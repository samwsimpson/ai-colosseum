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
from fastapi import Header, HTTPException

OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 3.1 - FIRESTORE) <<")
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY env var is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Prevent duplicate greetings on rapid reconnects for the same (user, conversation)
import time
RECENT_GREETS = {}  # key: (user_id, conversation_id) -> monotonic timestamp
GREETING_TTL_SECONDS = 10

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

# --- Conversations API ---

async def _user_from_bearer(authorization: str | None):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    # <-- use your existing JWT verify function here -->
    user = await verify_and_get_user(token)  # if your helper is sync, remove await
    if not user or not user.get("id"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/conversations")
async def list_conversations(user=Depends(get_current_user), limit: int = 30):    

    col = db.collection("conversations")
    # newest first
    q = col.where("user_id", "==", user["id"]).order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit)

    items = []
    async for doc in q.stream():
        d = doc.to_dict() or {}
        items.append({
            "id": doc.id,
            "title": d.get("title") or "New conversation",
            "updated_at": (d.get("updated_at") or d.get("created_at")),
        })
    return {"items": items}

@app.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: dict, user=Depends(get_current_user)):
    
    new_title = (body or {}).get("title", "")
    new_title = new_title.strip()[:120] or "Untitled"

    ref = db.collection("conversations").document(conv_id)
    # enforce ownership
    snap = await ref.get()
    if not snap.exists or (snap.to_dict() or {}).get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await ref.update({"title": new_title, "updated_at": datetime.utcnow()})
    return {"ok": True}

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
        conversation_count += 1


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
            httponly=True,  # Corrected typo: httpy_only -> httponly
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

        message_output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        user_input_queue: asyncio.Queue = asyncio.Queue()

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

        # --- randomized opening greeting (guarded: once per WS; skip if recently greeted) ---
        try:
            import random

            # Only greet if this conversation has no messages yet (prevents greeting on resumes/reconnects)
            has_any = False
            async for _ in (conv_ref.collection("messages").limit(1).stream()):
                has_any = True
                break

            # Also skip if we greeted this (user, conversation) very recently (e.g., quick reconnect)
            key = (user['id'], conv_ref.id)
            now = time.monotonic()
            last = RECENT_GREETS.get(key)

            if (not has_any) and (last is None or (now - last) > GREETING_TTL_SECONDS):
                greeter = random.choice(agent_names)  # e.g. "ChatGPT", "Claude", "Gemini", "Mistral"
                greeting_text = f"Hi {user_display_name}, what can we help you with?"

                # Send greeting message without a typing indicator
                queue_send_nowait({"sender": greeter, "text": greeting_text})
                
                # Persist so your history shows the opener
                await save_message(conv_ref, role="assistant", sender=greeter, content=greeting_text)

                RECENT_GREETS[key] = now

        except Exception as e:
            print("[opening greeting] skipped:", e)
        # --- end greeting ---

        def make_agent_system(name: str) -> str:
            base = {
                "ChatGPT": CHATGPT_SYSTEM,
                "Claude": CLAUDE_SYSTEM,
                "Gemini": GEMINI_SYSTEM,
                "Mistral": MISTRAL_SYSTEM,
            }.get(name, "You are an assistant.")

            # Use a single, simplified prompt for all agents
            base_prompt = (
                "You are in a group chat with a user and other AI assistants. "
                "Your primary goal is to address the user's request. "
                "Read and understand the full conversation history. "
                "Speak only when you have a distinct contribution to make or when directly addressed by the user. "
                "Be concise, direct, and avoid conversational fillers like 'As an AI...' or 'I apologize...' unless you are correcting a factual error. "
                "Do not comment on other assistants' turns or try to hand off the conversation. "
            )

            return (
                f"{base}\n\n"
                f"Your name is {name}. Participants: {user_display_name} (user), {', '.join(agent_names)} (assistants).\n"
                f"{base_prompt}"
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
            def __init__(self, name, llm_config, system_message,
                        message_output_queue: asyncio.Queue,
                        proxy_for_forward: "WebSocketUserProxyAgent"):
                super().__init__(name=name, llm_config=llm_config, system_message=system_message)
                self._message_output_queue = message_output_queue
                self._proxy_for_forward = proxy_for_forward

            async def a_receive(self, message, sender=None, request_reply=False, silent=False):
                # Just receive. Do NOT toggle typing here; only toggle in a_generate_reply when we are actually speaking.
                return await super().a_receive(message, sender=sender, request_reply=request_reply, silent=silent)


            async def a_generate_reply(self, messages=None, sender=None, **kwargs):
                # turn typing on for the assistant
                try:
                    self._message_output_queue.put_nowait({"sender": self.name, "typing": True, "text": ""})
                except Exception:
                    pass

                result = None
                out_text = ""

                try:
                    # SIMPLE RETRY (3 attempts with backoff) to avoid silent failures
                    delay = 0.8
                    last_exc = None
                    for attempt in range(3):
                        try:
                            result = await super().a_generate_reply(messages=messages, sender=sender, **kwargs)
                            break
                        except Exception as e:
                            last_exc = e
                            if attempt == 2:
                                raise
                            await asyncio.sleep(delay)
                            delay *= 2
                except Exception:
                    # If all retries fail, show a small, friendly bubble instead of silence
                    try:
                        self._message_output_queue.put_nowait({
                            "sender": self.name,
                            "text": "I hit a temporary issue and couldn’t reply. Please ask again."
                        })
                    except Exception:
                        pass
                    result = {"content": ""}  # unify return shape for caller
                else:
                    # Normalize to a plain string
                    if isinstance(result, dict):
                        out_text = (result.get("content") or result.get("text") or "").strip()
                    elif isinstance(result, str):
                        out_text = result.strip()
                    elif result is None:
                        out_text = ""
                    else:
                        out_text = str(result).strip()

                    

                finally:
                    # ALWAYS clear typing even if we error/return early
                    try:
                        self._message_output_queue.put_nowait({"sender": self.name, "typing": False, "text": ""})
                    except Exception:
                        pass

                return result

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
                # Keep this for de-duping, but remove the rest
                self._last_text_by_sender: dict[str, str] = {}
                # NEW: authoritative ledger for each assistant's latest number
                
                self._assistant_names_list = sorted(list(self._assistant_name_set))
            
            async def a_receive(self, message, sender=None, request_reply=True, silent=False):
                """
                Intercepts assistant->user messages and forwards them to the browser,
                while handling termination and basic de-duplication.
                """
                # ---- normalize the incoming payload ----
                speaker = (
                    getattr(sender, "name", None)
                    or (message.get("name") if isinstance(message, dict) else None)
                    or "Unknown"
                )
                raw_text = (
                    (message.get("content") if isinstance(message, dict) else None)
                    or (message.get("text") if isinstance(message, dict) else None)
                    or (str(message) if not isinstance(message, dict) else "")
                )
                text = (raw_text or "").strip()

                # Never show manager/system lines
                if speaker and speaker.lower().strip() in {"chat_manager", "manager", "orchestrator"}:
                    return {"content": "", "name": speaker}

                # De-dup identical consecutive messages (This is a helpful heuristic to keep)
                prev = self._last_text_by_sender.get(speaker)
                if prev and prev.strip() == text.strip():
                    return {"content": "", "name": speaker}
                self._last_text_by_sender[speaker] = text

                # Forward to browser
                if text:
                    payload = {"sender": speaker, "text": text}
                    try:
                        self._message_output_queue.put_nowait(payload)
                    except Exception:
                        await self._message_output_queue.put(payload)

                    # Send the typing-off message
                    try:
                        self._message_output_queue.put_nowait({"sender": speaker, "typing": False, "text": ""})
                    except Exception:
                        pass

                # Keep base behavior for Autogen bookkeeping
                return await super().a_receive(message, sender=sender, request_reply=request_reply, silent=silent)

            async def a_generate_reply(
                self,
                messages: List[Dict[str, Any]] | None = None,
                sender: autogen.ConversableAgent | None = None,
                **kwargs,
            ) -> Union[str, Dict, None]:
                # Block until a user message is received
                user_input = await self._user_input_queue.get()
                return {"content": user_input, "role": "user", "name": self.name}

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

        seed_messages = []
        if conv_doc.get("summary"):
            seed_messages.append({
                "role": "system",
                "content": f"Conversation summary so far:\n{conv_doc['summary']}"
            })

        seed_messages.append({
            "role": "system",
            "content": (
                f"Participants: {', '.join(agent_names)}. User: {user_display_name}. "
                "Memory: This conversation is persistent. Rely on the 'Conversation summary' for prior context."
            )
        })
        # Add this block right after the seed_messages block
        groupchat = autogen.GroupChat(
            agents=agents,
            messages=seed_messages,
            max_round=999999,
            speaker_selection_method="auto",
            allow_repeat_speaker=True,
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=chatgpt_llm_config,
            system_message=(
                "You are the group chat manager. Your role is to determine the next speaker based on the conversation history. "
                "Choose the agent most relevant to the user's last request. "
                "If the user addressed a specific agent by name, select that agent. "
                "If the user asked 'everyone', cycle through each assistant once in a random order. "
                "If the conversation turn is complete, select the 'User' to signal for their next input. "
                "Output only the name of the next speaker, for example, 'User' or 'ChatGPT'."
            )
        )
        
        # --- NEW CODE: Redesigned chat loop to prevent deadlock ---
        
        async def main_chat_loop(ws: WebSocket, proxy: WebSocketUserProxyAgent, manager, conv_ref):
            is_first_message = True
            chat_task: Optional[asyncio.Task] = None

            
            

            while True:
                try:
                    data = await ws.receive_json()
                    
                    if isinstance(data, dict) and data.get("type") in ("ping", "pong"):
                        if data.get("type") == "ping":
                            await ws.send_json({"type": "pong"})
                        continue
                        
                    user_message = data["message"]

                    await ws.send_json({"sender": proxy.name, "text": user_message})
                    await save_message(conv_ref, role="user", sender=proxy.name, content=user_message)
                    await maybe_set_title(conv_ref, user_message)

                    if is_first_message:
                        # Kick off the manager loop in the background so we never block WS reads
                        chat_task = asyncio.create_task(proxy.a_initiate_chat(manager, message=user_message))
                        is_first_message = False
                    else:
                        # Feed subsequent user turns into the running manager loop
                        await proxy.a_inject_user_message(user_message)


                except WebSocketDisconnect:
                    print("main_chat_loop: WebSocket disconnected.")
                    if chat_task and not chat_task.done():
                        chat_task.cancel()
                    break
                except Exception as e:
                    print(f"main_chat_loop error: {e}")
                    if chat_task and not chat_task.done():
                        chat_task.cancel()
                    break
                    
        # This task sends AI messages from the output queue to the WebSocket
        async def message_consumer_task(queue: asyncio.Queue, ws: WebSocket, conv_ref, agent_name_set: set, user_internal_name: str):
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=120)
                    await ws.send_json(msg)

                    sender = (msg or {}).get("sender") or "System"
                    text = (msg or {}).get("text") or (msg or {}).get("content") or ""

                    if sender == user_internal_name or not text:
                        continue

                    role = "assistant" if sender in agent_name_set else "system"
                    await save_message(conv_ref, role=role, sender=sender, content=text)
                    await maybe_refresh_summary(conv_ref)

                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect as e:
                    print(f"message_consumer_task: client disconnected (code={getattr(e, 'code', 'unknown')})")
                    break
                except Exception as e:
                    print(f"message_consumer_task error: {e}")
                    break
        
        async def keepalive_task(ws: WebSocket):
            try:
                while True:
                    await asyncio.sleep(20)
                    await ws.send_json({"sender": "System", "type": "server_ping"})
            except WebSocketDisconnect as e:
                print(f"keepalive_task: client disconnected (code={getattr(e, 'code', 'unknown')})")
            except Exception as e:
                print(f"keepalive_task error: {e}")
        
        # We start the tasks that are truly independent and long-running
        consumer_task_coro = asyncio.create_task(
            message_consumer_task(message_output_queue, websocket, conv_ref, agent_names, safe_user_name)
        )
        main_chat_loop_coro = asyncio.create_task(
            main_chat_loop(websocket, user_proxy, manager, conv_ref)
        )
        keepalive_task_coro = asyncio.create_task(keepalive_task(websocket))

        try:
            await asyncio.gather(
                consumer_task_coro,
                main_chat_loop_coro,
                keepalive_task_coro
            )
        except WebSocketDisconnect:
            print("WebSocket closed normally.")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"General exception in main gather: {e}\n{tb}")
            try:
                await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
            except Exception:
                pass
        finally:
            for t in (consumer_task_coro, main_chat_loop_coro, keepalive_task_coro):
                if not t.done():
                    t.cancel()
            await asyncio.gather(consumer_task_coro, main_chat_loop_coro, keepalive_task_coro, return_exceptions=True)

    except WebSocketDisconnect:
        print("Outer WebSocketDisconnect caught.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"General exception: {e}\n{tb}")
        try:
            await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
        except WebSocketDisconnect:
            pass
