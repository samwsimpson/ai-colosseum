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
import httpx
from jose import JWTError, jwt
from datetime import datetime, timedelta, date
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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/google-auth")

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
    initial_config = await websocket.receive_json()
    print("INITIAL CONFIG RECEIVED:", initial_config)
    message_output_queue = asyncio.Queue(maxsize=100)
    # ...safe_enqueue / queue_send...
    user_name = initial_config.get('user_name', 'User')
    sanitized_user_name = user_name.replace(" ", "_")
    agent_names = ["ChatGPT", "Claude", "Gemini", "Mistral"]

    def make_agent_system(name: str) -> str:
        base = {
            "ChatGPT":  CHATGPT_SYSTEM,
            "Claude":   CLAUDE_SYSTEM,
            "Gemini":   GEMINI_SYSTEM,
            "Mistral":  MISTRAL_SYSTEM,
        }.get(name, "You are an assistant.")
        return (
            f"{base}\n\n"
            f"Your name is {name}. Participants: {sanitized_user_name} (user), "
            f"{', '.join(agent_names)} (AIs). {roster_text}"
        )
    try:
        await websocket.accept()

        user = await get_current_user(token=token)
        
        user_doc = await db.collection('users').document(user['id']).get()
        user_data = user_doc.to_dict()
        
        user_subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
        user_subscription_data = user_subscription_doc.to_dict()

        if user_subscription_data['monthly_limit'] is not None:
            first_day_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            conversation_count_query = db.collection('conversations').where(
                'user_id', '==', user['id']
            ).where(
                'subscription_id', '==', user_data['subscription_id']
            ).where(
                'timestamp', '>=', first_day_of_month
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
        
        new_conversation_data = {
            'user_id': user['id'],
            'timestamp': datetime.utcnow(),
            'subscription_id': user_data['subscription_id'],
        }
        await db.collection('conversations').add(new_conversation_data)

        initial_config = await websocket.receive_json()
        print("INITIAL CONFIG RECEIVED:", initial_config)
        # Bounded queue to avoid blocking if the browser is slow
        message_output_queue = asyncio.Queue(maxsize=100)

        def safe_enqueue(payload: dict):
            """Best-effort enqueue without blocking; drop oldest if full."""
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
        user_name = initial_config.get('user_name', 'User')
        sanitized_user_name = user_name.replace(" ", "_")

        async def queue_send(queue: asyncio.Queue, payload):
            """Non-blocking put: if the queue is full, drop the oldest and enqueue the new payload."""
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = queue.get_nowait()  # drop oldest
                except Exception:
                    pass
                try:
                    queue.put_nowait(payload)
                except Exception:
                    pass


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
        
        class WebSocketAssistantAgent(autogen.AssistantAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._message_output_queue = message_output_queue
            
            async def a_send_typing_indicator(self, is_typing: bool):
                await queue_send(self._message_output_queue, {
                    "sender": self.name,
                    "typing": is_typing,
                    "text": ""
                })

            async def a_generate_reply(self, messages: List[Dict[str, Any]] = None, sender: autogen.ConversableAgent = None, **kwargs) -> Union[str, Dict, None]:
                await self.a_send_typing_indicator(is_typing=True)
                try:
                    reply = await super().a_generate_reply(messages=messages, sender=sender, **kwargs)
                finally:
                    await self.a_send_typing_indicator(is_typing=False)
                
                return reply

        class CustomSpeakerSelector:
            def __init__(self, agents, user_name):
                self.assistant_agents = [agent for agent in agents if agent.name not in [user_name, "System"]]
                self.multi_agent_reply_state = {"is_active": False, "agents_to_reply": []}
                self.agent_by_name = {agent.name: agent for agent in agents}
                self.message_history = []
                self.loop_detected = False
                self.waiting_on_user_to_break_loop = False
                self.user_name = user_name

            def __call__(self, last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
                last_message = groupchat.messages[-1]
                last_content = last_message["content"].strip().lower()
                self.message_history.append((last_speaker.name, last_content))
                self.message_history = self.message_history[-10:]

                if self.waiting_on_user_to_break_loop:
                    if last_speaker.name == self.user_name:
                        self.waiting_on_user_to_break_loop = False
                        self.loop_detected = False
                    return self.agent_by_name[self.user_name]

                if (
                    len(self.message_history) >= 4 and
                    self.message_history[-1] == self.message_history[-3] and
                    self.message_history[-2] == self.message_history[-4]
                ):
                    print("⚠️ Loop detected. Returning to User.")
                    self.loop_detected = True
                    self.waiting_on_user_to_break_loop = True

                    asyncio.create_task(
                        queue_send(
                            self.agent_by_name[self.user_name]._message_output_queue,
                            {"sender": "System", "text": "⚠️ Loop detected. Waiting for your input to continue the conversation..."}
                        )
                    )
                    return self.agent_by_name[self.user_name]
                
                if last_speaker.name in [a.name for a in self.assistant_agents]:
                    for agent in self.assistant_agents:
                        if f"{agent.name.lower()}, would you like" in last_content or f"{agent.name.lower()} would you like" in last_content:
                            print(f"Detected turn pass from {last_speaker.name} to {agent.name}.")
                            self.multi_agent_reply_state["is_active"] = False
                            return self.agent_by_name[agent.name]

                if last_speaker.name == self.user_name:
                    # Detect explicit addressing by name at start: e.g., "Claude, ...", "ChatGPT:", "Gemini - ..."
                    addressed = None
                    for agent in self.assistant_agents:
                        if re.match(rf"^\s*{re.escape(agent.name)}[\s,:-]", last_content, re.IGNORECASE):
                            addressed = agent.name
                            break

                    multi_agent_keywords = [
                        "everyone", "all of you", "both of you", "each of you",
                        "all agents", "all ais", "all models", "you all", "you guys",
                        "all bots", "the group"
                    ]

                    # If explicitly addressed, route to that agent
                    if addressed:
                        self.multi_agent_reply_state["is_active"] = False
                        return self.agent_by_name[addressed]

                    # If multi-agent broadcast, fan out to all agents (once each)
                    if any(word in last_content for word in multi_agent_keywords):
                        self.multi_agent_reply_state["is_active"] = True
                        self.multi_agent_reply_state["agents_to_reply"] = [agent.name for agent in self.assistant_agents]
                        return self.agent_by_name[self.multi_agent_reply_state["agents_to_reply"].pop(0)]

                    # Name mentioned anywhere (not necessarily at start) → prefer that agent
                    for agent in self.assistant_agents:
                        if agent.name.lower() in last_content:
                            self.multi_agent_reply_state["is_active"] = False
                            return self.agent_by_name[agent.name]

                    # Default
                    self.multi_agent_reply_state["is_active"] = False
                    return self.agent_by_name.get("ChatGPT", self.agent_by_name[self.user_name])


                elif last_speaker.name in [a.name for a in self.assistant_agents]:
                    if self.multi_agent_reply_state["is_active"]:
                        if self.multi_agent_reply_state["agents_to_reply"]:
                            return self.agent_by_name[self.multi_agent_reply_state["agents_to_reply"].pop(0)]
                        else:
                            self.multi_agent_reply_state["is_active"] = False
                            return self.agent_by_name[self.user_name]
                    
                    return self.agent_by_name[self.user_name]

                return self.agent_by_name[self.user_name]
        
        class WebSocketUserProxyAgent(autogen.UserProxyAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._user_input_queue = asyncio.Queue()
                self._message_output_queue = message_output_queue

            async def a_receive(self, message, sender, request_reply: bool = True, silent: bool = False):
                if isinstance(message, dict):
                    content = message.get("content", "")
                    sender_name = message.get("name", sender.name if sender else self.name)
                elif isinstance(message, str):
                    content = message
                    sender_name = sender.name if sender else self.name
                else:
                    return None
                
                cleaned = re.sub(r'(TERMINATE|Task Completed\.)[\s\S]*', '', content).strip()
                
                if cleaned:
                    await queue_send(self._message_output_queue, {"sender": sender_name, "text": cleaned})

                return None

            async def a_generate_reply(self, messages: List[Dict[str, Any]] = None, sender: autogen.ConversableAgent = None, **kwargs) -> Union[str, Dict, None]:
                new_user_message = await self._user_input_queue.get()
                return new_user_message

            async def a_inject_user_message(self, message: str):
                await self._user_input_queue.put(message)

        user_proxy = WebSocketUserProxyAgent(
            name=sanitized_user_name,
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


                # --- Roster & rules so every model knows who's here and who "you all" means ---
        agent_names = [a.name for a in agents if a.name != sanitized_user_name]
        roster_text = (
            "SYSTEM: Multi-agent room context\n"
            f"- USER: {user_name} (your messages appear as '{sanitized_user_name}')\n"
            f"- AGENTS PRESENT: {', '.join(agent_names)}\n\n"
            "Conversation rules for all agents:\n"
            "1) You are in a shared room with ALL listed agents. Treat their messages as visible context.\n"
            "2) Always prefix any references with the speaker name you are responding to when helpful, e.g.,\n"
            "   'ChatGPT → Claude:' or 'Claude → User:' if addressing someone directly.\n"
            "3) If the user addresses someone by name at the START of their message (e.g., 'Claude,' or 'ChatGPT:'), "
            "that named agent should respond first.\n"
            "4) If the user says 'you all', 'everyone', 'all agents', 'both of you', 'each of you', 'all of you', "
            "'you guys', or similar, each agent should respond ONCE, concisely.\n"
            "5) If you are not the addressed agent, but the user is clearly speaking to another agent, hold your reply "
            "unless explicitly invited.\n"
            "6) Use the user’s name when appropriate.\n"
            "Examples of addressing:\n"
            "- 'Claude, what do you think?' → Claude responds first.\n"
            "- 'ChatGPT and Gemini, please answer.' → ChatGPT then Gemini respond once each.\n"
            "- 'You all: one sentence each.' → Each listed agent replies once.\n"
            "- 'Claude do you see ChatGPT’s message?' → Claude responds; others hold.\n"
        )

        # Add roster/rules to each assistant's system prompt
        for a in agents:
            if isinstance(a, WebSocketAssistantAgent):
                try:
                    a.system_message = f"{a.system_message}\n\n{roster_text}"
                except Exception:
                    # Fallback: some autogen versions keep it under ._system_message
                    if hasattr(a, "_system_message"):
                        a._system_message = f"{getattr(a, '_system_message', '')}\n\n{roster_text}"


        # Also seed the group chat history with a system message containing the roster/rules.
        initial_messages = [{
            "role": "system",
            "name": "System",
            "content": roster_text
        }]



        print("AGENTS IN GROUPCHAT:")
        for a in agents:
            print(f" - {a.name}")
        if len(agents) < 2:
            await websocket.send_json({"sender": "System", "text": "No AIs available for your subscription."})
            return
        
        selector = CustomSpeakerSelector(agents, user_name=sanitized_user_name)

        groupchat = autogen.GroupChat(
            agents=agents,
            messages=initial_messages,  # seed with roster/rules
            max_round=999999,
            speaker_selection_method=selector,
            allow_repeat_speaker=True
        )

        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=False,
            system_message=GROUPCHAT_SYSTEM_MESSAGE
        )
        
        async def message_consumer_task(queue: asyncio.Queue, ws: WebSocket):
            while True:
                msg = await queue.get()
                try:
                    await ws.send_json(msg)
                except WebSocketDisconnect as e:
                    print(f"message_consumer_task: WebSocketDisconnect while sending (code={getattr(e, 'code', 'unknown')}).")
                    break
                except RuntimeError as e:
                    # Happens if the socket is closing/closed or event loop issues.
                    print(f"message_consumer_task runtime error: {e}")
                    break
                except Exception as e:
                    print(f"message_consumer_task unexpected send error: {e}")
                    break
        
        async def user_input_handler_task(ws: WebSocket, proxy: WebSocketUserProxyAgent):
            while True:
                try:
                    data = await ws.receive_json()

                    # --- heartbeat handling ---
                    if isinstance(data, dict):
                        t = data.get("type")
                        if t == "ping":
                            # reply and ignore
                            try:
                                await ws.send_json({"type": "pong"})
                            except Exception:
                                pass
                            continue
                        if t == "pong":
                            # ignore client pongs
                            continue
                    # --- end heartbeat handling ---

                    # Normal chat payloads only: dict with a non-empty string "message"
                    if isinstance(data, dict):
                        msg = data.get("message")
                        if isinstance(msg, str) and msg.strip():
                            await proxy.a_inject_user_message(msg)
                        else:
                            # Ignore anything that isn't a chat message
                            print(f"Ignoring non-chat payload: {data!r}")
                    else:
                        # Ignore non-dict payloads
                        print(f"Ignoring non-dict payload: {data!r}")

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error handling user input: {e}")
                    try:
                        await ws.send_json({"sender": "System", "text": f"Error: {e}"})
                    except WebSocketDisconnect:
                        break

        consumer_task = asyncio.create_task(message_consumer_task(message_output_queue, websocket))
        conversation_task = asyncio.create_task(
            user_proxy.a_initiate_chat(manager, message=initial_config.get('message', 'Hello!'))
        )
        input_handler_task = asyncio.create_task(user_input_handler_task(websocket, user_proxy))

        # Optional keepalive ping to avoid idle WS disconnects
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
            # Keep WS alive until client disconnects
            await input_handler_task
        finally:
            for task in (conversation_task, consumer_task, ka_task):
                task.cancel()
            await asyncio.gather(conversation_task, consumer_task, ka_task, return_exceptions=True)

    except WebSocketDisconnect:
        print("WebSocket closed")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"General exception: {e}\n{tb}")  # <-- full stack trace to logs
        try:
            await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
        except WebSocketDisconnect:
            pass
