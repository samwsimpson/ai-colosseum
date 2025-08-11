import sys
import traceback
print("TOP OF api.py: Script starting...", file=sys.stderr)
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import autogen
import asyncio
import re
from typing import List, Dict, Any, Union
from jose import JWTError, jwt
from datetime import datetime, timedelta
from google.auth.transport import requests as google_requests
from google.oauth2.id_token import verify_oauth2_token
import stripe
from google.cloud import firestore
from google_auth_oauthlib.flow import Flow

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 2.0 - FIRESTORE) <<")
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY env var is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# ==== System prompts (define once, before websocket handler) ====

def _env(name: str, default: str) -> str:
    # Use env var if present; otherwise fallback to a sane default
    val = os.getenv(name)
    return val if (val is not None and val.strip() != "") else default

CHATGPT_SYSTEM = _env(
    "CHATGPT_SYSTEM",
    "You are ChatGPT. Be concise, accurate, and helpful. When unsure, ask for clarification."
)

CLAUDE_SYSTEM = _env(
    "CLAUDE_SYSTEM",
    "You are Claude. Provide careful reasoning and clear explanations. Avoid hallucinations."
)

GEMINI_SYSTEM = _env(
    "GEMINI_SYSTEM",
    "You are Gemini. Answer succinctly, cite assumptions, and highlight uncertainties."
)

MISTRAL_SYSTEM = _env(
    "MISTRAL_SYSTEM",
    "You are Mistral. Give practical, straightforward answers with minimal fluff."
)

GROUPCHAT_SYSTEM_MESSAGE = _env(
    "GROUPCHAT_SYSTEM_MESSAGE",
    (
        "You are the group chat coordinator. Keep discussion focused, prevent loops, and ensure each agent only speaks "
        "when it adds value. If agents repeat themselves or stall, hand control back to the user."
    )
)
# ==== end system prompts ====


app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Re-added CORS middleware for local development
origins = [
    "http://localhost:3000",
    "https://aicolosseum.app",
    "https://www.aicolosseum.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/google-auth")

class ChatMessage(BaseModel):
    sender: str
    text: str

class GoogleAuthCode(BaseModel):
    code: str
    redirect_uri: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    user_id: str

# Initialize Firestore DB client without explicit credentials
db = firestore.AsyncClient()
print("FIRESTORE_CLIENT_INITIALIZED: db = firestore.AsyncClient()", file=sys.stderr)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        user_ref = db.collection('users').document(user_id)
        user_doc = await user_ref.get()

        if not user_doc.exists:
            raise credentials_exception
            
        user = user_doc.to_dict()
        user['id'] = user_doc.id
        return user
    except JWTError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.on_event("startup")
async def startup_event():
    print("STARTUP EVENT: Initializing Firestore...")
    try:
        subscriptions_ref = db.collection('subscriptions')
        print("STARTUP EVENT: subscriptions_ref created")
        plans = {
            'Free': {'monthly_limit': 5, 'price_id': 'free_price_id_placeholder'},
            'Starter': {'monthly_limit': 25, 'price_id': 'starter_price_id_placeholder'},
            'Pro': {'monthly_limit': 200, 'price_id': 'pro_price_id_placeholder'},
            'Enterprise': {'monthly_limit': None, 'price_id': 'enterprise_price_id_placeholder'},
        }
        print("STARTUP EVENT: Plans defined")
        
        for name, data in plans.items():
            print(f"STARTUP EVENT: Checking for subscription plan: {name}")
            doc_ref = subscriptions_ref.document(name)
            print(f"STARTUP EVENT: doc_ref for {name} created")
            doc = await doc_ref.get()
            print(f"STARTUP EVENT: doc.exists for {name}: {doc.exists}")
            if not doc.exists:
                print(f"STARTUP EVENT: Plan '{name}' not found. Creating it.")
                await doc_ref.set(data)
                print(f"Created subscription plan: {name}")
            else:
                print(f"Plan '{name}' already exists.")
        print("STARTUP EVENT: Firestore initialization complete.")
    except Exception as e:
        print(f"STARTUP EVENT: Failed to initialize Firestore collections: {e}")
        raise

@app.get("/api/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    user_doc = await db.collection('users').document(current_user['id']).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict() or {}

    subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
    subscription_data = subscription_doc.to_dict()

    return {
        "user_name": user_data['name'],
        "user_id": user_doc.id,              # use doc id, not a field
        "user_plan_name": subscription_doc.id
    }
    
@app.get("/api/users/me/usage")
async def get_user_usage(current_user: dict = Depends(get_current_user)):
    user_doc = await db.collection('users').document(current_user['id']).get()
    user_data = user_doc.to_dict()
    
    subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
    subscription_data = subscription_doc.to_dict()
    
    if subscription_data['monthly_limit'] is None:
        return {
            "monthly_usage": 0,
            "monthly_limit": None
        }

    # in /api/users/me/usage and in the WS handler where you count conversations
    first_day_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    monthly_usage_query = db.collection('conversations').where(
        'user_id', '==', current_user['id']
    ).where(
        'subscription_id', '==', user_data['subscription_id']
    ).where(
        'timestamp', '>=', first_day_of_month
    )
    
    monthly_usage = 0
    async for _ in monthly_usage_query.stream():
        monthly_usage += 1


    return {
        "monthly_usage": monthly_usage,
        "monthly_limit": subscription_data['monthly_limit']
    }

@app.post("/api/google-auth", response_model=Token)
async def google_auth(auth_code: GoogleAuthCode):
    try:
        client_config = {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [
                    "https://aicolosseum.app/sign-in",
                    "https://www.aicolosseum.app/sign-in",
                    "http://localhost:3000/sign-in"
                ],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email', 'openid'],
            redirect_uri=auth_code.redirect_uri
        )

        flow.fetch_token(code=auth_code.code)
        credentials = flow.credentials
        
        idinfo = verify_oauth2_token(credentials.id_token, google_requests.Request(), credentials.client_id)
        google_id = idinfo['sub']
        
        users_ref = db.collection('users')
        user_doc_query = users_ref.where('google_id', '==', google_id).limit(1).stream()
        user_list = [doc async for doc in user_doc_query]

        if not user_list:
            free_plan_doc = await db.collection('subscriptions').document('Free').get()
            
            new_user_ref = users_ref.document()
            user_data = {
                'google_id': google_id,
                'name': idinfo['name'],
                'email': idinfo['email'],
                'subscription_id': free_plan_doc.id,
            }
            await new_user_ref.set(user_data)
            user_id = new_user_ref.id
            print(f"New user created: {idinfo['name']} on Free plan.")
        else:
            user_id = user_list[0].id
            print(f"User found in database: {user_list[0].to_dict()['name']}")

        token = create_access_token({"sub": user_id}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer", "user_name": idinfo['name'], "user_id": user_id}

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")


# === STRIPE IMPLEMENTATION ===
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class SubscriptionRequest(BaseModel):
    price_id: str

@app.post("/api/create-checkout-session")
async def create_checkout_session(request: SubscriptionRequest, current_user: dict = Depends(get_current_user)):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": request.price_id,
                    "quantity": 1,
                },
            ],
            mode="subscription",
            success_url="https://aicolosseum.app/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://aicolosseum.app/cancel",
            customer_email=current_user['email'],
        )
        return {"id": checkout_session.id, "url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, stripe_webhook_secret
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # ← IMPORTANT: this must be OUTSIDE the except blocks
    if event["type"] == "checkout.session.completed":
        sess_obj = event["data"]["object"]
        customer_email = sess_obj.get("customer_email")

        # Re-retrieve with expand to access line_items safely
        session = stripe.checkout.Session.retrieve(
            sess_obj["id"],
            expand=["line_items"]
        )

        price_id = None
        try:
            price_id = session.line_items.data[0].price.id
        except Exception:
            # Optional: fall back to metadata if you set it during checkout
            price_id = (session.get("metadata") or {}).get("price_id")

        if not (customer_email and price_id):
            return {"status": "ignored", "reason": "missing email or price_id"}

        users_ref = db.collection('users')
        user_doc_query = users_ref.where('email', '==', customer_email).limit(1).stream()
        user_docs = [doc async for doc in user_doc_query]

        subscriptions_ref = db.collection('subscriptions')
        new_subscription_doc_query = subscriptions_ref.where('price_id', '==', price_id).limit(1).stream()
        new_subscription_list = [doc async for doc in new_subscription_doc_query]

        if user_docs and new_subscription_list:
            user_ref = users_ref.document(user_docs[0].id)
            await user_ref.update({'subscription_id': new_subscription_list[0].id})
            print(f"User {customer_email} successfully subscribed to the {new_subscription_list[0].id} plan.")

    return {"status": "success"}
@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        await websocket.accept()

        # ---- auth & monthly limit ----
        user = await get_current_user(token=token)

        user_doc = await db.collection('users').document(user['id']).get()
        user_data = user_doc.to_dict()

        user_subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
        user_subscription_data = user_subscription_doc.to_dict()

        if user_subscription_data['monthly_limit'] is not None:
            first_day_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            conversation_count_query = (
                db.collection('conversations')
                .where('user_id', '==', user['id'])
                .where('subscription_id', '==', user_data['subscription_id'])
                .where('timestamp', '>=', first_day_of_month)
            )
            conversation_count = 0
            async for _ in conversation_count_query.stream():
                conversation_count += 1

            if conversation_count >= user_subscription_data['monthly_limit']:
                await websocket.send_json({
                    "sender": "System",
                    "text": "Your monthly conversation limit has been reached. Please upgrade your plan to continue."
                })
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Limit reached")
                return

        await db.collection('conversations').add({
            'user_id': user['id'],
            'timestamp': datetime.utcnow(),
            'subscription_id': user_data['subscription_id'],
        })

        # ---- init from client ----
        initial_config = await websocket.receive_json()
        message_output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        def queue_send_nowait(payload: dict):
            try:
                message_output_queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = message_output_queue.get_nowait()  # drop oldest
                except Exception:
                    pass
                try:
                    message_output_queue.put_nowait(payload)
                except Exception:
                    pass

        user_name = initial_config.get('user_name', 'User') or 'User'
        user_display_name = user_name.replace('_', ' ').strip()  # show “Sam Simpson”, never underscores

        # Keep this list in one place
        agent_names = ["ChatGPT", "Claude", "Gemini", "Mistral"]

        roster_text = (
            "SYSTEM: Multi-agent room context\n"
            f"- USER: {user_display_name}\n"
            f"- AGENTS PRESENT: {', '.join(agent_names)}\n\n"
            "Conversation rules for all agents:\n"
            "1) You are in a shared room with ALL listed agents. Treat their messages as visible context.\n"
            "2) If the user addresses someone by name at the START of their message (e.g., 'Claude,' or 'ChatGPT:'), "
            "that named agent should respond first.\n"
            "3) If the user says 'you all', 'everyone', 'all agents', 'both of you', 'each of you', or similar, "
            "each agent should respond ONCE, concisely.\n"
            "4) If one assistant clearly addresses another assistant, let the addressee reply next.\n"
            "5) Mentions are NOT the same as addresses. Referencing an agent by name does not require that agent to reply.\n"
            "6) If the user does not name anyone, the last assistant who spoke should reply.\n"
            "7) When addressing another assistant directly, start with their name (e.g., 'Claude, ...'). "
            "When addressing the user, use natural language (e.g., 'Sam, ...'), not arrows or labels.\n"
        )

        def make_agent_system(name: str) -> str:
            base = {
                "ChatGPT": CHATGPT_SYSTEM,
                "Claude": CLAUDE_SYSTEM,
                "Gemini": GEMINI_SYSTEM,
                "Mistral": MISTRAL_SYSTEM,
            }.get(name, "You are an assistant.")
            return (
                f"{base}\n\n"
                f"Your name is {name}. Participants: {user_display_name} (user), {', '.join(agent_names)} (AIs).\n"
                f"{roster_text}"
            )

        # ---- model configs ----
        chatgpt_llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_type": "openai"
            }],
            "temperature": 0.5,
            "timeout": 90
        }
        claude_llm_config = {
            "config_list": [{
                "model": "claude-3-5-sonnet-20240620",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "api_type": "anthropic"
            }],
            "temperature": 0.7,
            "timeout": 90
        }
        gemini_llm_config = {
            "config_list": [{
                "model": "gemini-1.5-pro",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "api_type": "google"
            }],
            "temperature": 0.7,
            "timeout": 90
        }
        mistral_llm_config = {
            "config_list": [{
                "model": "mistral-large-latest",
                "api_key": os.getenv("MISTRAL_API_KEY"),
                "api_type": "mistral"
            }],
            "temperature": 0.7,
            "timeout": 90
        }

        # ---- agent classes ----
        class WebSocketAssistantAgent(autogen.AssistantAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._message_output_queue = message_output_queue

            async def a_send_typing_indicator(self, is_typing: bool):
                queue_send_nowait({"sender": self.name, "typing": is_typing, "text": ""})

            async def a_generate_reply(self, messages=None, sender=None, **kwargs):
                await self.a_send_typing_indicator(True)
                try:
                    return await super().a_generate_reply(messages=messages, sender=sender, **kwargs)
                finally:
                    await self.a_send_typing_indicator(False)

        class CustomSpeakerSelector:
            """
            - If an assistant directly addresses another assistant by name, the addressee replies next.
            - If the user replies without naming someone, route to the last assistant who spoke.
            - Mentions that are not direct addresses do not trigger a reply.
            - If the user says 'everyone'/'all of you', enable a one-pass multi-agent reply (each assistant once).
            - Basic loop detection hands control back to the user.
            """
            def __init__(self, agents, user_name):
                self.assistant_agents = [a for a in agents if a.name not in [user_name, "System"]]
                self.assistant_names = [a.name for a in self.assistant_agents]
                self.agent_by_name = {a.name: a for a in agents}
                self.user_name = user_name

                self.message_history = []      # (speaker, content_lower)
                self.previous_assistant = None # last assistant who actually spoke
                self.waiting_on_user_to_break_loop = False
                self.multi = {"active": False, "queue": []}

            def _broadcast_requested(self, text: str) -> bool:
                low = text.lower()
                return any(k in low for k in [
                    "everyone", "all of you", "both of you", "each of you",
                    "all agents", "all ais", "you all", "@all"
                ])

            def _direct_addressees(self, text: str):
                import re
                low = text.lower()
                addressees = []

                # At-start vocative: "Claude,", "Gemini:", "@Mistral —"
                for name in self.assistant_names:
                    nl = name.lower()
                    if re.search(rf'^\s*@?{re.escape(nl)}\b\s*[:,\-—]?', low):
                        addressees.append(name)

                # Handoff patterns anywhere
                for name in self.assistant_names:
                    nl = name.lower()
                    if re.search(rf'(?:over\s+to|hand\s+to|pass(?:\s+it)?\s+to)\s+{re.escape(nl)}\b', low):
                        if name not in addressees:
                            addressees.append(name)

                # Requesty vocatives near start
                req = r'(?:would you|can you|please|your (?:thoughts|take)|mind|could you)'
                for name in self.assistant_names:
                    nl = name.lower()
                    if re.search(rf'^\s*@?{re.escape(nl)}\b.*\b{req}\b', low):
                        if name not in addressees:
                            addressees.append(name)

                return addressees

            def __call__(self, last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
                # Guard: if no meaningful history, start with ChatGPT by default
                if not groupchat.messages:
                    return self.agent_by_name.get("ChatGPT", self.agent_by_name[self.user_name])  
                last_msg = groupchat.messages[-1]
                content = (last_msg.get("content") or "").strip()
                low = content.lower()

                # Loop detection
                self.message_history.append((last_speaker.name, low))
                self.message_history = self.message_history[-10:]

                if self.waiting_on_user_to_break_loop:
                    if last_speaker.name == self.user_name:
                        self.waiting_on_user_to_break_loop = False
                    return self.agent_by_name[self.user_name]

                if (
                    len(self.message_history) >= 4
                    and self.message_history[-1] == self.message_history[-3]
                    and self.message_history[-2] == self.message_history[-4]
                ):
                    self.waiting_on_user_to_break_loop = True
                    queue_send_nowait({"sender": "System", "text": "⚠️ Loop detected. Waiting for your input to continue..."})
                    return self.agent_by_name[self.user_name]

                # Assistant spoke last
                if last_speaker.name in self.assistant_names:
                    self.previous_assistant = self.agent_by_name[last_speaker.name]

                    # If they handed off, let the addressee speak
                    addressees = self._direct_addressees(content)
                    if addressees:
                        return self.agent_by_name[addressees[0]]

                    # Continue one-pass broadcast
                    if self.multi["active"]:
                        if self.multi["queue"]:
                            return self.agent_by_name[self.multi["queue"].pop(0)]
                        self.multi["active"] = False
                        return self.agent_by_name[self.user_name]

                    # Otherwise hand back to user
                    return self.agent_by_name[self.user_name]

                # User spoke last
                if last_speaker.name == self.user_name:
                    if self._broadcast_requested(content):
                        self.multi["active"] = True
                        self.multi["queue"] = [a.name for a in self.assistant_agents]
                        return self.agent_by_name[self.multi["queue"].pop(0)]

                    addressees = self._direct_addressees(content)
                    if addressees:
                        return self.agent_by_name[addressees[0]]

                    if self.previous_assistant is not None:
                        return self.previous_assistant

                    return self.assistant_agents[0] if self.assistant_agents else self.agent_by_name[self.user_name]

                return self.agent_by_name[self.user_name]

        class WebSocketUserProxyAgent(autogen.UserProxyAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._user_input_queue = asyncio.Queue()
                self._message_output_queue = message_output_queue

            async def a_receive(self, message, sender, request_reply: bool = True, silent: bool = False):
                # Normalize to string content
                if isinstance(message, dict):
                    content = message.get("content", "")
                elif isinstance(message, str):
                    content = message
                else:
                    return None

                # Prefer the actual assistant sender over the manager
                sender_name = None
                if sender is not None and getattr(sender, "name", None):
                    sender_name = sender.name
                    if sender_name.lower() in ("chat_manager", "groupchat_manager"):
                        sender_name = None

                # Fallback to a name inside the message dict (some backends set this)
                if sender_name is None and isinstance(message, dict):
                    candidate = message.get("name") or message.get("sender")
                    if candidate and str(candidate).strip():
                        sender_name = str(candidate).strip()

                # Final fallback
                if sender_name is None:
                    sender_name = "System"

                # Strip any trailing termination hints Autogen sometimes includes
                cleaned = re.sub(r'(TERMINATE|Task Completed\.)[\s\S]*', '', content).strip()

                if cleaned:
                    await self._message_output_queue.put({"sender": sender_name, "text": cleaned})

                return None

            async def a_generate_reply(self, messages=None, sender=None, **kwargs):
                return await self._user_input_queue.get()

            async def a_inject_user_message(self, message: str):
                await self._user_input_queue.put(message)

        # ---- build roster ----
        user_proxy = WebSocketUserProxyAgent(
            name=user_display_name,
            human_input_mode="NEVER",
            message_output_queue=message_output_queue,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda x: isinstance(x, dict) and x.get("content", "").endswith("TERMINATE")
        )

        agents = [user_proxy]
        agents.append(WebSocketAssistantAgent("ChatGPT", llm_config=chatgpt_llm_config, system_message=make_agent_system("ChatGPT"), message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Claude",  llm_config=claude_llm_config,  system_message=make_agent_system("Claude"),  message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Gemini",  llm_config=gemini_llm_config,  system_message=make_agent_system("Gemini"),  message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Mistral", llm_config=mistral_llm_config, system_message=make_agent_system("Mistral"), message_output_queue=message_output_queue))

        print("AGENTS IN GROUPCHAT:")
        for a in agents:
            print(f" - {a.name}")
        if len(agents) < 2:
            await websocket.send_json({"sender": "System", "text": "No AIs available for your subscription."})
            return

        selector = CustomSpeakerSelector(agents, user_name=user_display_name)

        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[{"role": "system", "name": "System", "content": roster_text}],
            max_round=999999,
            speaker_selection_method=selector,
            allow_repeat_speaker=True
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=False,
            system_message=GROUPCHAT_SYSTEM_MESSAGE
        )

        # ---- tasks ----
        async def message_consumer_task(queue: asyncio.Queue, ws: WebSocket):
            while True:
                msg = await queue.get()
                try:
                    await ws.send_json(msg)
                except WebSocketDisconnect as e:
                    print(f"message_consumer_task: client disconnected (code={getattr(e, 'code', 'unknown')})")
                    break
                except Exception as e:
                    print(f"message_consumer_task error: {e}")
                    break

        async def user_input_handler_task(ws: WebSocket, proxy: WebSocketUserProxyAgent):
            while True:
                try:
                    data = await ws.receive_json()
                    if isinstance(data, dict):
                        t = data.get("type")
                        if t == "ping":
                            await ws.send_json({"type": "pong"})
                            continue
                        if t == "pong":
                            continue
                        if "message" in data and isinstance(data["message"], str):
                            await proxy.a_inject_user_message(data["message"])
                            continue
                    # ignore unexpected payloads
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"user_input_handler_task error: {e}")
                    try:
                        await ws.send_json({"sender": "System", "text": f"Error: {e}"})
                    except WebSocketDisconnect:
                        break

        consumer_task = asyncio.create_task(message_consumer_task(message_output_queue, websocket))
        conversation_task = asyncio.create_task(user_proxy.a_initiate_chat(manager, message=initial_config.get('message', 'Hello!')))
        input_handler_task = asyncio.create_task(user_input_handler_task(websocket, user_proxy))

        async def keepalive_task(ws: WebSocket):
            try:
                while True:
                    await asyncio.sleep(20)
                    await ws.send_json({"sender": "System", "type": "ping"})
            except WebSocketDisconnect as e:
                print(f"keepalive_task: client disconnected (code={getattr(e, 'code', 'unknown')})")
            except Exception as e:
                print(f"keepalive_task error: {e}")

        ka_task = asyncio.create_task(keepalive_task(websocket))

        try:
            await asyncio.gather(
                conversation_task,
                input_handler_task,
                consumer_task,
                ka_task,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"WebSocket tasks error: {e}\n{tb}")
            try:
                await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
            except Exception:
                pass
        finally:
            for t in (conversation_task, input_handler_task, consumer_task, ka_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(conversation_task, input_handler_task, consumer_task, ka_task, return_exceptions=True)


    except WebSocketDisconnect:
        print("WebSocket closed")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"General exception: {e}\n{tb}")
        try:
            await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
        except WebSocketDisconnect:
            pass
