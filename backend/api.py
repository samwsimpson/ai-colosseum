import sys
import traceback
import time
import random
import re
import asyncio
import jwt as pyjwt
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, constr
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from google.auth.transport import requests as google_requests
from google.oauth2.id_token import verify_oauth2_token
from google.cloud import firestore
from google.cloud.firestore_v1 import Increment
import stripe
from google_auth_oauthlib.flow import Flow
from openai import AsyncOpenAI
import autogen

print("TOP OF api.py: Script starting...", file=sys.stderr)

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

RECENT_GREETS = {}
GREETING_TTL_SECONDS = 10

def _env(name: str, default: str) -> str:
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

async def call_with_retry(op_coro_factory, ws, *, retries: int = 2, base_delay: float = 0.8):
    for attempt in range(retries + 1):
        try:
            return await op_coro_factory()
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "service_tier_capacity_exceeded" in msg or "capacity" in msg.lower()
            if not is_429:
                raise
            try:
                await ws.send_json({"sender": "System", "text": "Provider is under heavy load, retrying..."})
            except Exception:
                pass
            if attempt == retries:
                try:
                    await ws.send_json({"sender": "System", "text": "Still busy. Please try again in a moment."})
                except Exception:
                    pass
                raise
            delay = base_delay * (2 ** attempt) + random.random() * 0.3
            await asyncio.sleep(delay)

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

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

db = firestore.AsyncClient()
print("FIRESTORE_CLIENT_INITIALIZED: db = firestore.AsyncClient()", file=sys.stderr)

async def get_or_create_conversation(user_id: str, initial_config: dict):
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
    if not content:
        return
    now = datetime.now(timezone.utc)
    await conv_ref.collection("messages").add({
        "role": role,
        "sender": sender,
        "content": content,
        "timestamp": now
    })
    await conv_ref.update({
        "updated_at": now,
        "message_count": Increment(1),
    })

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
        "user_id": user_doc.id,
        "user_plan_name": subscription_doc.id
    }

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
    msgs.reverse()
    return {
        "id": conv_id,
        "title": conv.get("title") or "Conversation",
        "summary": conv.get("summary") or "",
        "message_count": conv.get("message_count", len(msgs)),
        "created_at": _ts_iso(conv.get("created_at")),
        "updated_at": _ts_iso(conv.get("updated_at")),
        "messages": msgs,
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
        access_token = create_access_token({"sub": user_id}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
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
        return ""
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

TITLE_DEFAULT = "New conversation"

async def maybe_set_title(conv_ref, user_text: str):
    if not user_text:
        return
    try:
        snap = await conv_ref.get()
        doc = snap.to_dict() or {}
        current = (doc.get("title") or "").strip()
        if current and current != TITLE_DEFAULT:
            return
        trimmed = user_text.strip().replace("\n", " ")
        if 0 < len(trimmed) <= 60:
            title = trimmed.rstrip(".?!")
        else:
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
    try:
        conv_snap = await conv_ref.get()
        conv = conv_snap.to_dict() or {}
        mc = int(conv.get("message_count", 0))
        lsc = int(conv.get("last_summary_count", 0))
        if (mc - lsc) < threshold:
            return
        msgs = []
        q = (conv_ref.collection("messages")
             .order_by("timestamp", direction=firestore.Query.DESCENDING)
             .limit(window))
        async for doc in q.stream():
            d = doc.to_dict() or {}
            msgs.append(d)
        msgs.reverse()
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
        prompt = (
            "Summarize this chat so far in 8-12 concise bullet points, "
            "capture decisions, to-dos, names, and key facts. Keep neutral tone.\n\n"
            f"{transcript}"
        )
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
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    if event["type"] == "checkout.session.completed":
        sess_obj = event["data"]["object"]
        customer_email = sess_obj.get("customer_email")
        session = stripe.checkout.Session.retrieve(
            sess_obj["id"],
            expand=["line_items"]
        )
        price_id = None
        try:
            price_id = session.line_items.data[0].price.id
        except Exception:
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

# --- Main WebSocket Endpoint (Completely Rewritten) ---
@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()

    try:
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

        initial_config = await websocket.receive_json()
        conv_ref, conv_doc = await get_or_create_conversation(user['id'], {
            **(initial_config or {}),
            "subscription_id": user_data.get("subscription_id"),
        })
        await websocket.send_json({"sender": "System", "type": "conversation_id", "id": conv_ref.id})
        
        message_output_queue = asyncio.Queue(maxsize=100)
        user_input_queue = asyncio.Queue()

        raw_user_name = (initial_config.get('user_name') or 'User').strip()
        safe_user_name = re.sub(r'[^A-Za-z0-9_-]', '_', raw_user_name) or 'User'
        user_display_name = raw_user_name.replace('_', ' ').strip()
        agent_names = ["ChatGPT", "Claude", "Gemini", "Mistral"]
        
        # --- Randomized opening greeting ---
        try:
            has_any = False
            async for _ in (conv_ref.collection("messages").limit(1).stream()):
                has_any = True
                break
            key = (user['id'], conv_ref.id)
            now = time.monotonic()
            last = RECENT_GREETS.get(key)
            if (not has_any) and (last is None or (now - last) > GREETING_TTL_SECONDS):
                greeter = random.choice(agent_names)
                greeting_text = f"Hi {user_display_name}, what can we help you with?"
                await websocket.send_json({"sender": greeter, "text": greeting_text})
                await save_message(conv_ref, role="assistant", sender=greeter, content=greeting_text)
                RECENT_GREETS[key] = now
        except Exception as e:
            print(f"[opening greeting] skipped due to error: {e}")

        # --- Agent setup ---
        def get_all_agents():
            return [a for a in agents if a.name in agent_names]
            
        def last_speaker_selection(last_speaker, groupchat):
            messages = groupchat.messages
            last_message_content = messages[-1]['content'].lower() if messages else ""
            
            # Check for "everyone"
            if "everyone" in last_message_content:
                eligible_agents = [a for a in get_all_agents() if a.name != last_speaker.name]
                random.shuffle(eligible_agents)
                return eligible_agents
            
            # Check for specific agent mention
            for agent_name in agent_names:
                if agent_name.lower() in last_message_content:
                    for agent in get_all_agents():
                        if agent.name == agent_name:
                            return agent
            
            # Default to last speaker, or a random agent if no last speaker
            if last_speaker and last_speaker.name in agent_names:
                return last_speaker
            
            return random.choice(get_all_agents())

        chatgpt_llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}], "temperature": 0.5, "timeout": 90}
        claude_llm_config = {"config_list": [{"model": "claude-3-5-sonnet-20240620", "api_key": os.getenv("ANTHROPIC_API_KEY")}], "temperature": 0.7, "timeout": 90}
        gemini_llm_config = {"config_list": [{"model": "gemini-1.5-pro", "api_key": os.getenv("GEMINI_API_KEY")}], "temperature": 0.7, "timeout": 90}
        mistral_llm_config = {"config_list": [{"model": "mistral-large-latest", "api_key": os.getenv("MISTRAL_API_KEY")}], "temperature": 0.7, "timeout": 90}

        class WebSocketAssistantAgent(autogen.AssistantAgent):
            def __init__(self, name, llm_config, system_message, message_output_queue: asyncio.Queue):
                super().__init__(name=name, llm_config=llm_config, system_message=system_message)
                self._message_output_queue = message_output_queue
            
            async def a_send(self, message, recipient, request_reply=None, silent=False, **kwargs):
                if not silent and message.get("content"):
                    await self._message_output_queue.put({"sender": self.name, "typing": True, "text": ""})
                    await asyncio.sleep(1) # simulate typing
                    await self._message_output_queue.put({"sender": self.name, "text": message["content"]})
                    await self._message_output_queue.put({"sender": self.name, "typing": False, "text": ""})
                
                return await super().a_send(message, recipient, request_reply=request_reply, silent=silent, **kwargs)

        class WebSocketUserProxyAgent(autogen.UserProxyAgent):
            def __init__(self, name, message_output_queue, user_input_queue, assistant_name_set):
                super().__init__(name=name, human_input_mode="NEVER", is_termination_msg=lambda x: isinstance(x, dict) and x.get("content", "").endswith("TERMINATE"))
                self._message_output_queue = message_output_queue
                self._user_input_queue = user_input_queue
                self._assistant_name_set = assistant_name_set

            async def a_get_human_input(self, prompt, **kwargs):
                return await self._user_input_queue.get()
                
            async def a_send(self, message, recipient, request_reply=None, silent=False, **kwargs):
                if not silent and message.get("content"):
                    # This echoes the user's message back to the UI.
                    # It's not a real AI response, but it completes the local message loop.
                    await self._message_output_queue.put({"sender": self.name, "text": message["content"]})
                return await super().a_send(message, recipient, request_reply=request_reply, silent=silent, **kwargs)

        agents = []
        user_proxy = WebSocketUserProxyAgent(
            name=safe_user_name,
            message_output_queue=message_output_queue,
            user_input_queue=user_input_queue,
            assistant_name_set=set(agent_names)
        )
        agents.append(user_proxy)
        
        for name, system_prompt, llm_config in [
            ("ChatGPT", make_agent_system("ChatGPT"), chatgpt_llm_config),
            ("Claude", make_agent_system("Claude"), claude_llm_config),
            ("Gemini", make_agent_system("Gemini"), gemini_llm_config),
            ("Mistral", make_agent_system("Mistral"), mistral_llm_config),
        ]:
            agents.append(WebSocketAssistantAgent(
                name=name,
                llm_config=llm_config,
                system_message=system_prompt,
                message_output_queue=message_output_queue
            ))

        print("AGENTS IN GROUPCHAT:")
        for a in agents:
            print(f" - {a.name}")
        if len(agents) < 2:
            await websocket.send_json({"sender": "System", "text": "No AIs available for your subscription."})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="No agents")
            return

        groupchat = autogen.GroupChat(
            agents=agents,
            messages=seed_messages,
            max_round=999999,
            speaker_selection_method=last_speaker_selection,
            allow_repeat_speaker=True,
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=chatgpt_llm_config,
            system_message="You are the group chat manager. Your goal is to manage the conversation flow.",
        )
        
        # --- NEW CODE: Redesigned chat loop to prevent deadlock ---
        async def main_chat_loop(ws: WebSocket, proxy: WebSocketUserProxyAgent, manager, conv_ref):
            autogen_task = None
            is_first_message = True
            
            while True:
                try:
                    data = await ws.receive_json()
                    if isinstance(data, dict) and data.get("type") in ("ping", "pong"):
                        if data.get("type") == "ping":
                            await ws.send_json({"type": "pong"})
                        continue
                    
                    user_message = data.get("message", "")
                    if not user_message:
                        continue

                    await save_message(conv_ref, role="user", sender=proxy.name, content=user_message)
                    await maybe_set_title(conv_ref, user_message)

                    if is_first_message:
                        autogen_task = asyncio.create_task(proxy.a_initiate_chat(manager, message=user_message))
                        is_first_message = False
                    else:
                        await proxy.a_inject_user_message(user_message)
                
                except WebSocketDisconnect:
                    print("main_chat_loop: WebSocket disconnected.")
                    if autogen_task and not autogen_task.done():
                        autogen_task.cancel()
                    break
                except Exception as e:
                    print(f"main_chat_loop error: {e}")
                    if autogen_task and not autogen_task.done():
                        autogen_task.cancel()
                    break

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
        except asyncio.CancelledError:
            print("Tasks were cancelled, shutting down.")
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

