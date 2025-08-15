import sys
import traceback
print("TOP OF api.py: Script starting...", file=sys.stderr)
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi import Request, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import autogen
import asyncio
import re
from typing import List, Dict, Any, Union, Optional
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from google.auth.transport import requests as google_requests
from google.oauth2.id_token import verify_oauth2_token
from google.cloud.firestore_v1 import Increment
from google.cloud import firestore  # for Query.DESCENDING
import stripe
from google_auth_oauthlib.flow import Flow
from openai import AsyncOpenAI
from collections import defaultdict

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
        "when it adds value. If agents repeat themselves or stall, hand control back to the user. "
        "Do not allow repetitive salutations: agents should greet at most once at the start of a new conversation; "
        "on follow-up turns they should answer directly without re-greeting."
    )
)

# ==== end system prompts ====

# ===== Retry & fallback helpers for LLM calls =====
import random

async def call_with_retry(op_coro_factory, ws, *, retries: int = 2, base_delay: float = 0.8):
    """
    Run an awaitable produced by op_coro_factory() with exponential backoff on 429 / capacity errors.
    Sends lightweight status updates to the websocket so the UI isn't blank.
    """
    for attempt in range(retries + 1):
        try:
            return await op_coro_factory()
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "service_tier_capacity_exceeded" in msg or "capacity" in msg.lower()
            if not is_429:
                raise  # not a capacity issue, bubble up

            # Tell the UI what's going on
            try:
                await ws.send_json({"sender": "System", "text": "Provider is under heavy load, retrying..."})
            except Exception:
                pass

            if attempt == retries:
                # Give a final friendly message then re-raise so logs show it
                try:
                    await ws.send_json({"sender": "System", "text": "Still busy. Please try again in a moment."})
                except Exception:
                    pass
                raise

            # jittered backoff
            delay = base_delay * (2 ** attempt) + random.random() * 0.3
            await asyncio.sleep(delay)


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
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception

        user_ref = db.collection('users').document(user_id)
        user_doc = await user_ref.get()
        if not user_doc.exists:
            raise credentials_exception

        user = user_doc.to_dict() or {}
        user['id'] = user_doc.id
        return user
    except pyjwt.PyJWTError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(*, data: dict, expires_delta: timedelta = timedelta(days=14)) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire, "type": "refresh"})
    return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)



# --- Conversation persistence helpers ---

async def get_or_create_conversation(user_id: str, initial_config: dict):
    """
    Returns (conv_ref, conv_doc_dict).
    If initial_config['conversation_id'] is present -> reuse it.
    Else if initial_config['resume_last'] is truthy -> resume user's most recent conversation.
    Else create a new conversation.
    """
    cfg = initial_config or {}
    conv_id = cfg.get("conversation_id")
    resume_last = bool(cfg.get("resume_last"))
    now = datetime.utcnow()

    conversations = db.collection("conversations")

    if conv_id:
        conv_ref = conversations.document(conv_id)
        await conv_ref.set({"user_id": user_id, "updated_at": now}, merge=True)
        doc = await conv_ref.get()
        return conv_ref, (doc.to_dict() or {})

    if resume_last:
        try:
            q = (
                conversations.where("user_id", "==", user_id)
                .order_by("updated_at", direction=firestore.Query.DESCENDING)
                .limit(1)
            )
            last = [doc async for doc in q.stream()]
            if last:
                conv_ref = conversations.document(last[0].id)
                await conv_ref.update({"updated_at": now})
                doc = await conv_ref.get()
                return conv_ref, (doc.to_dict() or {})
        except Exception as e:
            print("Resume-last query failed; creating new conversation. Error:", e)

    conv_ref = conversations.document()
    await conv_ref.set(
        {
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "subscription_id": cfg.get("subscription_id"),
            "title": cfg.get("title") or "New conversation",
            "summary": "",
        }
    )
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

# === Conversations REST ===
from typing import Optional, List
from pydantic import BaseModel, constr
from google.cloud import firestore

def _ts_iso(v):
    try:
        return v.isoformat()
    except Exception:
        return None

class RenameBody(BaseModel):
    title: constr(min_length=1, max_length=120)

@app.get("/api/conversations")
async def list_conversations(user=Depends(get_current_user)):
    items: List[dict] = []
    q = (
        db.collection("conversations")
        .where("user_id", "==", user["id"])
        .order_by("updated_at", direction=firestore.Query.DESCENDING)
        .limit(100)
    )
    async for d in q.stream():
        c = d.to_dict() or {}
        items.append({
            "id": d.id,
            "title": c.get("title") or "New conversation",
            "updated_at": _ts_iso(c.get("updated_at")),
            "message_count": c.get("message_count", 0),
            "summary": c.get("summary", ""),
        })
    return {"items": items}


@app.get("/api/conversations/{conv_id}/messages")
async def list_messages(conv_id: str, limit: int = 50, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    conv = await conv_ref.get()
    if (not conv.exists) or ((conv.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = []
    q = (conv_ref.collection("messages")
         .order_by("timestamp", direction=firestore.Query.DESCENDING)
         .limit(limit))
    async for m in q.stream():
        d = m.to_dict() or {}
        msgs.append({
            "id": m.id,
            "role": d.get("role"),
            "sender": d.get("sender"),
            "content": d.get("content"),
            "timestamp": _ts_iso(d.get("timestamp")),
        })
    msgs.reverse()
    return {"items": msgs}


@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: RenameBody, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    snap = await conv_ref.get()
    if (not snap.exists) or ((snap.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")
    await conv_ref.update({"title": body.title, "updated_at": datetime.now(timezone.utc)})
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    snap = await conv_ref.get()
    if (not snap.exists) or ((snap.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")

    while True:
        batch = db.batch()
        count = 0
        async for m in conv_ref.collection("messages").limit(300).stream():
            batch.delete(m.reference)
            count += 1
        if count == 0:
            break
        await batch.commit()

    await conv_ref.delete()
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    snap = await conv_ref.get()
    if (not snap.exists) or ((snap.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = snap.to_dict() or {}

    msgs = []
    q = conv_ref.collection("messages").order_by("timestamp")
    async for m in q.stream():
        d = m.to_dict() or {}
        msgs.append({
            "role": d.get("role"),
            "sender": d.get("sender"),
            "content": d.get("content"),
            "timestamp": _ts_iso(d.get("timestamp")),
        })

    return {
        "id": conv_id,
        "title": conv.get("title") or "Conversation",
        "summary": conv.get("summary") or "",
        "message_count": conv.get("message_count", len(msgs)),
        "created_at": _ts_iso(conv.get("created_at")),
        "updated_at": _ts_iso(conv.get("updated_at")),
        "messages": msgs,
    }

# === end Conversations REST ===


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
async def google_auth(auth_code: GoogleAuthCode, response: Response):
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
            scopes=[
                "https://www.googleapis.com/auth/userinfo.profile",
                "https://www.googleapis.com/auth/userinfo.email",
                "openid",
            ],
            redirect_uri=auth_code.redirect_uri,
        )
        flow.fetch_token(code=auth_code.code)
        credentials = flow.credentials

        idinfo = verify_oauth2_token(credentials.id_token, google_requests.Request(), credentials.client_id)
        google_id = idinfo["sub"]

        users_ref = db.collection("users")
        q = users_ref.where("google_id", "==", google_id).limit(1).stream()
        docs = [doc async for doc in q]

        if not docs:
            free_plan_doc = await db.collection("subscriptions").document("Free").get()
            new_user_ref = users_ref.document()
            user_data = {
                "google_id": google_id,
                "name": idinfo.get("name"),
                "email": idinfo.get("email"),
                "subscription_id": free_plan_doc.id,
            }
            await new_user_ref.set(user_data)
            user_id = new_user_ref.id
            user_name = user_data["name"] or "User"
        else:
            user_id = docs[0].id
            data = docs[0].to_dict() or {}
            user_name = data.get("name") or "User"

        # Access token (front-end uses this)
        access_token = create_access_token({"sub": user_id}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

        # Refresh token cookie (silent refresh later)
        refresh_token = create_refresh_token(data={"sub": user_id})
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            max_age=14 * 24 * 60 * 60,
            httponly=True,
            secure=True,
            samesite="none",
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            user_name=user_name,
            user_id=user_id,
        )

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")



# === STRIPE IMPLEMENTATION ===
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class SubscriptionRequest(BaseModel):
    price_id: str

@app.post("/api/refresh")
async def refresh_access_token(request: Request):
    rt = request.cookies.get("refresh_token")
    if not rt:
        raise HTTPException(status_code=401, detail="No refresh token")

    try:
        payload = pyjwt.decode(rt, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token payload")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_expires_in = 60 * 60
    new_access = create_access_token(
        data={"sub": user_id},
        expires_delta=timedelta(seconds=access_expires_in),
    )
    expires_at = int(datetime.now(timezone.utc).timestamp()) + access_expires_in
    return {"token": new_access, "expires_at": expires_at}



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
        # (No summary banner here — the UI can show titles via /api/conversations)

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
                f"{roster_text}\n"
                "Etiquette: Do not repeat greetings on every turn. Greet at most once when the conversation is new; "
                "for follow-up questions answer directly without re-greeting."
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
            def __init__(
                self,
                *args,
                message_output_queue: asyncio.Queue,
                proxy_for_forward: "WebSocketUserProxyAgent",
                **kwargs
            ):
                super().__init__(*args, **kwargs)
                self._message_output_queue = message_output_queue
                # <-- keep a handle to the user proxy so we can forward replies
                self._proxy_for_forward = proxy_for_forward

            async def a_send_typing_indicator(self, is_typing: bool):
                # (keep your typing indicators as-is)
                try:
                    # using the existing helper to enqueue a typing event
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: None
                    )  # noop; just ensures we're in async context
                finally:
                    # your queue helper already handles backpressure, we reuse it:
                    try:
                        # this uses the existing closure queue_send_nowait
                        queue_send_nowait({"sender": self.name, "typing": is_typing, "text": ""})
                    except Exception:
                        pass

            async def a_generate_reply(self, messages=None, sender=None, **kwargs):
                # show "is typing"
                await self.a_send_typing_indicator(True)
                try:
                    # let the model produce the reply
                    result = await super().a_generate_reply(messages=messages, sender=sender, **kwargs)

                    # normalize to a plain string for forwarding through the proxy
                    if isinstance(result, dict):
                        out_text = (result.get("content") or result.get("text") or "").strip()
                    elif isinstance(result, str):
                        out_text = result.strip()
                    else:
                        out_text = str(result).strip()

                    if out_text:
                        # Forward THROUGH the user proxy so your greeting/dup filters run
                        # (this is what ultimately writes to message_output_queue)
                        try:
                            await self._proxy_for_forward.a_receive(
                                {"content": out_text, "name": self.name},
                                sender=self,
                                request_reply=False,
                                silent=True,
                            )
                        except Exception as e:
                            print(f"[Assistant forward -> proxy] error: {e}")

                    # return the original result to the manager
                    return result
                finally:
                    # stop "is typing"
                    await self.a_send_typing_indicator(False)


        class CustomSpeakerSelector:
            """
            - If the user addresses an assistant by name (leading vocative like 'Claude,' or 'to Claude'), that assistant replies.
            - If the user gives no name, the last assistant who spoke replies.
            - If an assistant clearly hands off to another assistant, the addressee replies next.
            - If user asks 'everyone'/'you all', run a one-pass round (each assistant once), then hand control back to the user.
            """

            def __init__(self, agents: List[autogen.Agent], user_name: str):
                self.user_name = user_name
                self.agent_by_name = {a.name: a for a in agents}
                # assistants are everyone except the user/System
                self.assistant_names = [a.name for a in agents if a.name not in (user_name, "System")]
                self.assistant_agents = [self.agent_by_name[n] for n in self.assistant_names]
                self.previous_assistant: "Optional[autogen.Agent]" = None
                self.multi = {"active": False, "queue": []}

            def note_speaker(self, name: str) -> None:
                """Let the proxy tell us which assistant just spoke."""
                if name in self.assistant_names:
                    self.previous_assistant = self.agent_by_name.get(name, self.previous_assistant)

            def _broadcast_requested(self, text: str) -> bool:
                low = (text or "").lower()
                return any(k in low for k in (
                    "everyone", "all of you", "both of you", "each of you",
                    "all agents", "all ais", "you all", "@all"
                ))

            def _direct_addressees(self, text: str):
                """
                Returns a list of assistant names that the user directly addressed.
                Much more permissive:
                - "Name ..."  (no punctuation required)
                - "@Name", "Name:", "Name —", "Name -", "Name,"
                - "hey Name", "hi Name", "ok Name" at start
                - "over to Name", "hand to Name", "pass to Name" anywhere
                """
                low = (text or "").lower()
                addressees = []

                # Accept either self.assistant_names or a cached set
                names = getattr(self, "assistant_names", None) or list(getattr(self, "_assistant_name_set", [])) or []
                name_set = {n for n in names}

                # 1) Pure at-start vocative, no punctuation required
                #    e.g. "mistral where did that come from?"
                for name in names:
                    nl = name.lower()
                    if re.match(rf"^\s*@?{re.escape(nl)}\b", low):
                        addressees.append(name)

                # 2) Friendly openers at the very beginning: "hey name", "hi name", "ok name"
                if not addressees:
                    for name in names:
                        nl = name.lower()
                        if re.match(rf"^\s*(?:hey|hi|hello|ok|okay)\s+@?{re.escape(nl)}\b", low):
                            addressees.append(name)

                # 3) Handoff phrases anywhere
                for name in names:
                    nl = name.lower()
                    if re.search(rf"(?:over\s+to|hand\s+to|pass(?:\s+it)?\s+to)\s+{re.escape(nl)}\b", low):
                        if name not in addressees:
                            addressees.append(name)

                return addressees


            def __call__(self, last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
                # If no history at all, start with ChatGPT (or first assistant)
                if not groupchat.messages:
                    return self.agent_by_name.get("ChatGPT", self.assistant_agents[0])

                # Walk backwards to find the last *significant* message:
                # - skip manager/system lines
                # - prefer a real assistant or the user
                idx = len(groupchat.messages) - 1
                last_name = ""
                last_role = ""
                content = ""
                while idx >= 0:
                    msg = groupchat.messages[idx]
                    n = str(msg.get("name") or "")
                    r = str(msg.get("role") or "")
                    c = str(msg.get("content") or "")
                    if n.lower() not in ("chat_manager", "manager", "groupchatmanager") and r != "system":
                        last_name, last_role, content = n, r, c
                        break
                    idx -= 1

                # If we somehow found nothing usable, default to ChatGPT
                if not last_name and not last_role:
                    return self.agent_by_name.get("ChatGPT", self.assistant_agents[0])

                # A) Assistant spoke last → maybe handoff, else give floor back to user
                if last_name in self.assistant_names:
                    self.previous_assistant = self.agent_by_name[last_name]

                    # Did that assistant explicitly hand off to someone?
                    targets = self._direct_addressees(content)
                    if targets:
                        self.previous_assistant = self.agent_by_name[targets[0]]
                        return self.previous_assistant

                    # If we were mid “everyone” round, continue it
                    if self.multi["active"]:
                        if self.multi["queue"]:
                            return self.agent_by_name[self.multi["queue"].pop(0)]
                        # end of broadcast → back to user
                        self.multi["active"] = False
                        return self.agent_by_name[self.user_name]

                    # Normal case after an assistant: hand back to the user
                    return self.agent_by_name[self.user_name]

                # B) User spoke last → pick addressed agent, else last assistant, else first assistant
                if last_role == "user" or last_name == self.user_name:
                    if self._broadcast_requested(content):
                        self.multi["active"] = True
                        self.multi["queue"] = [a.name for a in self.assistant_agents]
                        return self.agent_by_name[self.multi["queue"].pop(0)]

                    targets = self._direct_addressees(content)
                    if targets:
                        return self.agent_by_name[targets[0]]

                    if self.previous_assistant is not None:
                        return self.previous_assistant

                    return self.assistant_agents[0] if self.assistant_agents else self.agent_by_name[self.user_name]

                # C) Fallback (e.g., a stray system/manager ended up “last”): ChatGPT or first assistant
                return self.agent_by_name.get("ChatGPT", self.assistant_agents[0])



        
     
        class WebSocketUserProxyAgent(autogen.UserProxyAgent):
            """
            User proxy that:
            - forwards every visible assistant message to the browser
            - suppresses repeated greeting spam (allow exactly one greeting per assistant per conversation)
            - lets us inject user text asynchronously
            - robustly determines the real speaker (ChatGPT/Claude/Gemini/Mistral)
            """
            def __init__(self, *args, message_output_queue: asyncio.Queue, assistant_name_set: set[str], **kwargs):
                super().__init__(*args, **kwargs)
                self._message_output_queue = message_output_queue
                self._user_input_queue: asyncio.Queue[str] = asyncio.Queue()
                self._assistant_name_set = set(assistant_name_set)
                self._assistant_name_lower = {n.lower(): n for n in self._assistant_name_set}
                self._utter_count: dict[str, int] = defaultdict(int)  # count messages per assistant
                self._greeted_once: set[str] = set()  # track which assistants have already greeted                
                self._last_text_by_sender: dict[str, str] = {}            # to drop exact repeats
                self._last_ai_speaker: str | None = None                  # who spoke last (AI)
                self._selector = None                                     # set in attach_selector()
                self._any_ai_ever: bool = False  # once any assistant speaks, drop pure greetings thereafter
                self._user_display_name = re.sub(r'[_-]+', ' ', (self.name or '')).strip() or 'there'

            def attach_selector(self, selector) -> None:
                self._selector = selector     
                               
            
            async def a_receive(self, message, sender=None, request_reply=True, silent=False):
                """
                Intercepts assistant->user messages, removes repetitive greetings,
                and forwards to the browser via message_output_queue.
                """
                import re

                # --- normalize incoming ---
                speaker = getattr(sender, "name", None) or (message.get("name") if isinstance(message, dict) else None) or "Unknown"
                raw_text = (
                    (message.get("content") if isinstance(message, dict) else None)
                    or (message.get("text") if isinstance(message, dict) else None)
                    or (str(message) if not isinstance(message, dict) else "")
                )
                text = (raw_text or "").strip()

                # --- helper: detect/strip greeting ---
                def strip_greeting(t: str) -> tuple[bool, str]:
                    """returns (was_greeting, stripped_text). If entire message is a greeting or
                    leading greeting phrase can be removed, it will be stripped."""
                    low = t.strip().lower()
                    # short, classic openings
                    looks_opening = bool(re.match(r'^(hi|hello|hey)\b', low))
                    mentions_help = any(k in low for k in ["assist you", "help you", "how can i"])
                    mentions_user = bool(self._user_display_name and re.search(
                        rf'\b{re.escape(self._user_display_name.lower())}\b', low
                    ))

                    # a greeting if it starts with hi/hello/hey and either mentions help OR user name
                    is_greeting = looks_opening and (mentions_help or mentions_user)

                    if not is_greeting:
                        return (False, t)

                    # strip the greeting prefix if there is anything substantive after it
                    # e.g., "Hi Sam! How can I assist you today? Here's an update…" -> "Here's an update…"
                    pat = rf'^(hi|hello|hey)\b(?:\s+{re.escape(self._user_display_name.lower())})?\W*(?:how can i(?:\s+\w+)?\s+(?:help|assist)\s+you(?:\s+\w+)?\??)?\W*'
                    stripped = re.sub(pat, '', t, flags=re.IGNORECASE).strip()
                    return (True, stripped)

                was_greet, stripped = strip_greeting(text)

                # If this speaker already greeted before, suppress pure greetings entirely
                if was_greet and speaker in self._greeted_once and not stripped:
                    # drop this message silently
                    return {"content": "", "name": speaker}

                # First time we see a greeting from this speaker, remember it.
                if was_greet and speaker not in self._greeted_once:
                    self._greeted_once.add(speaker)

                # Use stripped text if any; if stripping consumed everything, leave original (so first hello still shows once)
                if stripped:
                    text = stripped

                # --- forward to browser as before ---
                payload = {"sender": speaker, "text": text}
                try:
                    self._message_output_queue.put_nowait(payload)
                except Exception:
                    await self._message_output_queue.put(payload)

                # Let the base class continue (so the manager’s flow isn’t broken)
                return await super().a_receive(message, sender=sender, request_reply=request_reply, silent=silent)





            async def a_generate_reply(
                self,
                messages: List[Dict[str, Any]] | None = None,
                sender: autogen.ConversableAgent | None = None,
                **kwargs,
            ) -> Union[str, Dict, None]:
                # Wait for next user input injected from the websocket handler
                new_user_message = await self._user_input_queue.get()
                return new_user_message

            async def a_inject_user_message(self, message: str):
                await self._user_input_queue.put(message)



        

        # ---- build roster ----
        user_proxy = WebSocketUserProxyAgent(
            name=safe_user_name,  # <-- use the safe internal name
            human_input_mode="NEVER",
            message_output_queue=message_output_queue,
            assistant_name_set=set(agent_names), 
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda x: isinstance(x, dict) and x.get("content", "").endswith("TERMINATE")
        )
     
        agents = [user_proxy]
        agents.append(WebSocketAssistantAgent(
            "ChatGPT",
            llm_config=chatgpt_llm_config,
            system_message=make_agent_system("ChatGPT"),
            message_output_queue=message_output_queue,
            proxy_for_forward=user_proxy,
        ))
        agents.append(WebSocketAssistantAgent(
            "Claude",
            llm_config=claude_llm_config,
            system_message=make_agent_system("Claude"),
            message_output_queue=message_output_queue,
            proxy_for_forward=user_proxy,
        ))
        agents.append(WebSocketAssistantAgent(
            "Gemini",
            llm_config=gemini_llm_config,
            system_message=make_agent_system("Gemini"),
            message_output_queue=message_output_queue,
            proxy_for_forward=user_proxy,
        ))
        agents.append(WebSocketAssistantAgent(
            "Mistral",
            llm_config=mistral_llm_config,
            system_message=make_agent_system("Mistral"),
            message_output_queue=message_output_queue,
            proxy_for_forward=user_proxy,
        ))

        print("AGENTS IN GROUPCHAT:")
        for a in agents:
            print(f" - {a.name}")
        if len(agents) < 2:
            await websocket.send_json({"sender": "System", "text": "No AIs available for your subscription."})
            return

        selector = CustomSpeakerSelector(agents, user_name=safe_user_name)
        user_proxy.attach_selector(selector)

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
            speaker_selection_method=selector,  # <-- use your CustomSpeakerSelector
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
                # Track last assistant who actually spoke; the selector uses this.
                if role == "assistant":
                    try:
                        selector.previous_assistant = selector.agent_by_name.get(sender, selector.previous_assistant)
                    except Exception:
                        pass                    

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
                    await call_with_retry(
                        lambda: proxy.a_inject_user_message(user_message),
                        ws,
                        retries=2,
                        base_delay=0.8,
                    )

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
            call_with_retry(
                lambda: user_proxy.a_initiate_chat(manager, message=initial_config['message']),
                websocket,
                retries=2,
                base_delay=0.8,
            )
        )
        input_handler_task = asyncio.create_task(
            user_input_handler_task(websocket, user_proxy, conv_ref)
        )



        async def keepalive_task(ws: WebSocket):
            try:
                while True:
                    await asyncio.sleep(20)
                    await ws.send_json({"sender": "System", "type": "server_ping"})
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
