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
from datetime import datetime, timedelta, timezone
from google.auth.transport import requests as google_requests
from google.oauth2.id_token import verify_oauth2_token
from google.cloud.firestore_v1 import Increment
from google.cloud import firestore  # for Query.DESCENDING
import stripe
from google_auth_oauthlib.flow import Flow
from openai import AsyncOpenAI

OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
    "You are ChatGPT. Be concise, accurate, and helpful. When unsure, ask for clarification.\n\n"
    "Memory: This conversation is persistent across sessions. When the user references earlier sessions, "
    "rely on the 'Conversation summary' system message that may be provided at the start. "
    "If critical details seem missing, ask the user if you'd like to retrieve details from earlier turns."

)

CLAUDE_SYSTEM = _env(
    "CLAUDE_SYSTEM",
    "You are Claude. Provide careful reasoning and clear explanations. Avoid hallucinations.\n\n"
    "Memory: This conversation is persistent across sessions. When the user references earlier sessions, "
    "rely on the 'Conversation summary' system message that may be provided at the start. "
    "If critical details seem missing, ask the user if you'd like to retrieve details from earlier turns."    
)

GEMINI_SYSTEM = _env(
    "GEMINI_SYSTEM",
    "You are Gemini. Answer succinctly, cite assumptions, and highlight uncertainties.\n\n"
    "Memory: This conversation is persistent across sessions. When the user references earlier sessions, "
    "rely on the 'Conversation summary' system message that may be provided at the start. "
    "If critical details seem missing, ask the user if you'd like to retrieve details from earlier turns."    
)

MISTRAL_SYSTEM = _env(
    "MISTRAL_SYSTEM",
    "You are Mistral. Give practical, straightforward answers with minimal fluff.\n\n"
    "Memory: This conversation is persistent across sessions. When the user references earlier sessions, "
    "rely on the 'Conversation summary' system message that may be provided at the start. "
    "If critical details seem missing, ask the user if you'd like to retrieve details from earlier turns."    
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
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Conversation persistence helpers ---

async def get_or_create_conversation(user_id: str, initial_config: dict):
    """
    Returns (conv_ref, conv_doc_dict). If initial_config['conversation_id'] is present,
    it reuses it; otherwise creates a new conversation doc.
    """
    conv_id = (initial_config or {}).get("conversation_id")
    now = datetime.now(timezone.utc)

    conversations = db.collection("conversations")
    if conv_id:
        conv_ref = conversations.document(conv_id)
        # Ensure doc exists and bump updated_at
        await conv_ref.set({
            "user_id": user_id,
            "updated_at": now
        }, merge=True)
    else:
        conv_ref = conversations.document()
        await conv_ref.set({
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "subscription_id": (initial_config or {}).get("subscription_id"),
            "title": (initial_config or {}).get("title") or "New conversation",
            "summary": "",                 # running summary (optional)
            "message_count": 0,            # <- new
            "last_summary_count": 0,       # <- new
            "last_summary_at": None,       # <- optional
        })


    doc = await conv_ref.get()
    return conv_ref, (doc.to_dict() or {})

async def save_message(conv_ref, role: str, sender: str, content: str):
    """
    role: 'user' | 'assistant' | 'system'
    """
    if not content:
        return
    now = datetime.now(timezone.utc)
    await conv_ref.collection("messages").add({
        "role": role,
        "sender": sender,
        "content": content,
        "timestamp": now
    })
    # atomic increment; also advances updated_at
    await conv_ref.update({
        "updated_at": now,
        "message_count": Increment(1),
    })

# --- End Conversation persistence helpers ---

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
    first_day_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    monthly_usage_query = db.collection('conversations').where(
        'user_id', '==', current_user['id']
    ).where(
        'subscription_id', '==', user_data['subscription_id']
    ).where(
        'created_at', '>=', first_day_of_month
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

async def cheap_summarize(prompt: str) -> str:
    if openai_client is None:
        return ""   # silently skip if not configured
    try:
        resp = await openai_client.chat.completions.create(
            model=OPENAI_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You produce terse, accurate summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[cheap_summarize] error: {e}")
        return ""

# --- Conversation title helper ---
TITLE_DEFAULT = "New conversation"

async def maybe_set_title(conv_ref, user_text: str):
    """
    If the conversation does not have a real title yet, derive one from the user's text.
    Fast path uses the user's first short utterance; fallback asks cheap_summarize.
    """
    if not user_text:
        return
    try:
        snap = await conv_ref.get()
        doc = snap.to_dict() or {}
        current = (doc.get("title") or "").strip()
        if current and current != TITLE_DEFAULT:
            return  # already titled

        trimmed = user_text.strip().replace("\n", " ")
        # quick heuristic: short inputs become the title directly (no trailing punctuation)
        if 0 < len(trimmed) <= 60:
            title = trimmed.rstrip(".?!")
        else:
            # cheap model to generate a concise 6–8 word title
            prompt = (
                "Make a short chat title (max 8 words) for this user's request. "
                "Only output the title, no quotes or extra text.\n\n"
                f"Request: {user_text}"
            )
            title = await cheap_summarize(prompt)

        title = (title or "").strip()[:80] or TITLE_DEFAULT
        await conv_ref.update({"title": title})
    except Exception as e:
        print(f"maybe_set_title error: {e}")


async def maybe_refresh_summary(conv_ref, threshold: int = 10, window: int = 40):
    """
    Refresh the running summary only if at least `threshold` new messages
    have arrived since the last summary. Summarize only the latest `window` messages.
    """
    try:
        conv_snap = await conv_ref.get()
        conv = conv_snap.to_dict() or {}
        mc = int(conv.get("message_count", 0))
        lsc = int(conv.get("last_summary_count", 0))

        if (mc - lsc) < threshold:
            return  # nothing to do

        # pull last `window` messages, newest first
        msgs = []
        q = (conv_ref.collection("messages")
             .order_by("timestamp", direction=firestore.Query.DESCENDING)
             .limit(window))
        async for doc in q.stream():
            d = doc.to_dict() or {}
            msgs.append(d)
        msgs.reverse()  # chronological (oldest -> newest)

        # build a compact text for summarization
        lines = []
        for m in msgs:
            role = m.get("role") or "assistant"
            sender = m.get("sender") or "assistant"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{sender} ({role}): {content}")
        transcript = "\n".join(lines)

        if not transcript:
            return

        # --- call your cheapest summarizer (you can swap models here) ---
        prompt = (
            "Summarize this chat so far in 8-12 concise bullet points, "
            "capture decisions, to-dos, names, and key facts. Keep neutral tone.\n\n"
            f"{transcript}"
        )

        # Example using your OpenAI client safely (adjust to your client var)
        summary_text = await cheap_summarize(prompt)

        now = datetime.now(timezone.utc)
        await conv_ref.update({
            "summary": summary_text,
            "last_summary_count": mc,
            "last_summary_at": now,
            "updated_at": now,
        })
    except Exception as e:
        print(f"[maybe_refresh_summary] skipped due to error: {e}")



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
            first_day_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)          
            conversation_count_query = (
                db.collection('conversations')
                .where('user_id', '==', user['id'])
                .where('subscription_id', '==', user_data['subscription_id'])
                .where('created_at', '>=', first_day_of_month)
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

        # ---- init from client ----
        initial_config = await websocket.receive_json()

        # Create or reuse a conversation (persists across reconnects)
        conv_ref, conv_doc = await get_or_create_conversation(user['id'], {
            **(initial_config or {}),
            "subscription_id": user_data.get("subscription_id"),
        })

        # Tell the client which conversation id we’re using
        await websocket.send_json({
            "sender": "System",
            "type": "conversation_id",
            "id": conv_ref.id,
        })
        # If we already have a running summary, tell the client (for the banner)
        if (conv_doc or {}).get("summary"):
            await websocket.send_json({
                "sender": "System",
                "type": "context_summary",
                "summary": conv_doc["summary"],
            })
        # Optional: seed title from the very first greeting if it's meaningful
        try:
            seed_msg = (initial_config or {}).get("message", "").strip()
            if seed_msg and seed_msg.lower() not in ("hi", "hello", "hey"):
                await maybe_set_title(conv_ref, seed_msg)
        except Exception:
            pass

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

        raw_user_name = (initial_config.get('user_name') or 'User').strip()
        safe_user_name = re.sub(r'[^A-Za-z0-9_-]', '_', raw_user_name) or 'User'  # <-- no spaces/specials for the LLM API
        user_display_name = raw_user_name.replace('_', ' ').strip()               # <-- pretty name for your UI

        # Keep this list in one place
        agent_names = ["ChatGPT", "Claude", "Gemini", "Mistral"]

        roster_text = (
            "SYSTEM: Multi-agent room context\n"
            f"- USER: {user_display_name}\n"
            f"- AGENTS PRESENT: {', '.join(agent_names)}\n\n"
            "Conversation rules for all agents:\n"
            "1) You are in a shared room with ALL listed agents. Treat their messages as visible context.\n"
            "2) If the user addresses someone by name at the START of their message (e.g., 'Claude,', 'hey Gemini', 'mistral' or 'ChatGPT:'), "
            "that named agent should respond first.\n"
            "3) If the user says 'you all', 'everyone', 'all agents', 'both of you', 'each of you', or similar, "
            "each agent should respond ONCE, concisely.\n"
            "4) If one assistant clearly addresses another assistant, let the addressee reply next.\n"
            "5) Mentioning an assistant’s name does NOT always mean addressing them (it might be a reference). Prefer direct-address cues (leading name, or 'to <name>').\n"
            "6) If the user replies without naming anyone, assume they’re talking to the last assistant who spoke.\n"
            "7) When addressing another assistant directly, start with their name (e.g., 'Claude, ...'). "
            "When addressing the user, use natural language (e.g., 'Sam, ...'), not arrows or labels.\n"
            "8) Do not lecture about roles/identities unless the user asks. No meta: do not write lines like 'Sam → Claude:' or 'Claude → chat_manager:'.\n"
            "9) Keep replies helpful and concise. If greeted (e.g., 'Hi everyone'), reply with a short greeting and one clarifying question to move forward.\n"
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
                # Normalize incoming
                if isinstance(message, dict):
                    content = message.get("content", "")
                    incoming_name = message.get("name") or message.get("sender")
                elif isinstance(message, str):
                    content = message
                    incoming_name = None
                else:
                    return None

                cleaned = re.sub(r'(TERMINATE|Task Completed\.)[\s\S]*', '', content).strip()

                # Choose a display name that hides the manager
                sender_name = None
                if sender is not None and getattr(sender, "name", None):
                    if sender.name.lower() not in ("chat_manager", "groupchat_manager"):
                        sender_name = sender.name
                if sender_name is None and incoming_name:
                    sender_name = str(incoming_name).strip()
                if not sender_name:
                    sender_name = "System"

                # Push to the frontend
                if cleaned:
                    # Use the same non-blocking queue helper used by the assistants
                    queue_send_nowait({"sender": sender_name, "text": cleaned})

                # IMPORTANT: also persist this message in Autogen's conversation history
                # so other agents see it as context on their next turn.
                if isinstance(message, dict):
                    msg_for_history = {**message, "content": cleaned}
                else:
                    msg_for_history = cleaned

                return await super().a_receive(
                    msg_for_history,
                    sender,
                    request_reply=False,  # don't trigger a reply; just record it
                    silent=True           # don't echo anything extra
                )

            async def a_generate_reply(self, messages: List[Dict[str, Any]] = None, sender: autogen.ConversableAgent = None, **kwargs) -> Union[str, Dict, None]:
                new_user_message = await self._user_input_queue.get()
                return new_user_message

            async def a_inject_user_message(self, message: str):
                await self._user_input_queue.put(message)

        # ---- build roster ----
        user_proxy = WebSocketUserProxyAgent(
            name=safe_user_name,  # <-- use the safe internal name
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

        selector = CustomSpeakerSelector(agents, user_name=safe_user_name)

        # --- Seed messages: summary (if any) + memory primer and group rules ---
        seed_messages = []

        # If you already saved a running summary on the conversation, drop it in
        if conv_doc.get("summary"):
            seed_messages.append({
                "role": "system",
                "content": f"Conversation summary so far:\n{conv_doc['summary']}"
            })

        # Memory primer + participants and addressing rules
        participants = ", ".join(agent_names)  # just the AIs
        seed_messages.append({
            "role": "system",
            "content": (
                f"Participants: {participants}. User: {user_display_name}. Conversation ID: {conv_ref.id}.\n"
                "Memory: This conversation is persistent across sessions. Use the 'Conversation summary' (above) "
                "as ground truth for prior context. If a detail seems missing, briefly ask the user before assuming.\n\n"
                "Addressing rules for group chat:\n"
                "1) If the user starts with a name (e.g., 'Claude,'), that named agent should answer first.\n"
                "2) If no name is given and the user is replying to the last speaker, that last speaker should respond.\n"
                "3) Mentioning an agent by name in third-person does not imply that agent is being addressed.\n"
                "4) If the user says 'you all' or 'everyone', each agent may respond once (no duplicates), then yield.\n"
                "Keep responses concise; avoid name→name echoing."
            )
        })

        groupchat = autogen.GroupChat(
            agents=agents,
            messages=seed_messages,
            max_round=999999,
            speaker_selection_method=selector,
            allow_repeat_speaker=True,
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=False,
            system_message=GROUPCHAT_SYSTEM_MESSAGE
        )

        # ---- tasks ----
        async def message_consumer_task(queue: asyncio.Queue, ws: WebSocket, conv_ref, agent_name_set: set, user_internal_name: str):
            while True:
                msg = await queue.get()

                # forward to browser
                try:
                    await ws.send_json(msg)
                except WebSocketDisconnect as e:
                    print(f"message_consumer_task: client disconnected (code={getattr(e, 'code', 'unknown')})")
                    break
                except Exception as e:
                    print(f"message_consumer_task error: {e}")
                    break

                # persist (only if there's text)
                sender = (msg or {}).get("sender") or "System"
                text = (msg or {}).get("text") or ""
                if not text:
                    continue

                if sender in agent_name_set:
                    role = "assistant"
                elif sender == user_internal_name:
                    role = "user"
                else:
                    role = "system"

                try:
                    await save_message(conv_ref, role=role, sender=sender, content=text)
                    await maybe_refresh_summary(conv_ref)
                    
                except Exception as e:
                    # don't crash the socket if Firestore hiccups
                    print(f"save_message failed: {e}")
       
        async def user_input_handler_task(ws: WebSocket, proxy: WebSocketUserProxyAgent, conv_ref):
            while True:
                try:
                    data = await ws.receive_json()

                    # --- heartbeat handling ---
                    if isinstance(data, dict) and data.get("type") in ("ping", "pong"):
                        if data.get("type") == "ping":
                            # reply and ignore
                            await ws.send_json({"type": "pong"})
                        continue
                    # --- end heartbeat handling ---

                    user_message = data["message"]

                    # persist the user message
                    await save_message(conv_ref, role="user", sender=proxy.name, content=user_message)

                    # set a title if we don't have one yet
                    await maybe_set_title(conv_ref, user_message)

                    # feed into the group chat
                    await proxy.a_inject_user_message(user_message)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error handling user input: {e}")
                    try:
                        await ws.send_json({"sender": "System", "text": f"Error: {e}"})
                    except WebSocketDisconnect:
                        break

       
        assistant_name_set = set(agent_names)

        consumer_task = asyncio.create_task(
            message_consumer_task(message_output_queue, websocket, conv_ref, assistant_name_set, safe_user_name)
        )
        conversation_task = asyncio.create_task(
            user_proxy.a_initiate_chat(manager, message=initial_config.get('message', 'Hello!'))
        )
        input_handler_task = asyncio.create_task(
            user_input_handler_task(websocket, user_proxy, conv_ref)
        )



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
