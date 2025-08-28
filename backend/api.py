import sys
import traceback
print("TOP OF api.py: Script starting...", file=sys.stderr)
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi import Request, Response
from fastapi import Header
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
from google.auth.transport.requests import Request as AuthRequest
from google.auth import iam
from google.auth import default as google_auth_default
from google_auth_oauthlib.flow import Flow
from google.oauth2.id_token import verify_oauth2_token
from google.cloud.firestore_v1 import SERVER_TIMESTAMP, ArrayUnion, Increment
from google.cloud import firestore  # for Query.DESCENDING
from google.cloud import storage
from starlette.websockets import WebSocketState
import secrets

from werkzeug.utils import secure_filename
import mimetypes
import aiohttp
# --- Firestore client (async) ---
# Cloud Run provides default credentials, so no explicit key is required.
# Your code uses `await` on Firestore calls, so use AsyncClient (not the sync Client).
db = firestore.AsyncClient()
print("FIRESTORE_CLIENT_INITIALIZED: created firestore.AsyncClient()", file=sys.stderr)

import stripe

from openai import AsyncOpenAI
from collections import defaultdict


load_dotenv()
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
storage_client = storage.Client()

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 3.1 - FIRESTORE) <<")

# --- SECRET_KEY resolver: prod-safe, dev-friendly ---
def _resolve_secret_key() -> str:
    # 1) Plain env var
    env_val = (os.getenv("SECRET_KEY") or "").strip()
    if env_val:
        return env_val

    # 2) Optional: file-based secret (e.g., mounted secret or volume)
    key_file = os.getenv("SECRET_KEY_FILE")
    if key_file:
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                file_val = f.read().strip()
                if file_val:
                    return file_val
        except Exception as e:
            print(f"WARNING: Failed to read SECRET_KEY_FILE={key_file}: {e}", file=sys.stderr)

    # 3) Optional: Google Secret Manager (lazy import; only if name provided)
    sm_name = os.getenv("SECRET_KEY_SECRET_NAME")
    if sm_name and os.getenv("K_SERVICE"):  # only try GSM when running on Cloud Run
        try:
            from google.cloud import secretmanager  # lazy import to avoid hard dep if unused
            client = secretmanager.SecretManagerServiceClient()
            # support both ".../secrets/<name>" and full ".../versions/<ver>"
            if "/versions/" not in sm_name:
                sm_name = f"{sm_name}/versions/latest"
            resp = client.access_secret_version(name=sm_name)
            sm_val = resp.payload.data.decode("utf-8").strip()
            if sm_val:
                return sm_val
        except Exception as e:
            print(f"ERROR: Secret Manager fetch failed for SECRET_KEY: {e}", file=sys.stderr)

    # 4) Dev-only fallback when NOT on Cloud Run
    if not os.getenv("K_SERVICE"):
        tmp = secrets.token_urlsafe(64)
        print("WARNING: SECRET_KEY not set; using ephemeral DEV key", file=sys.stderr, flush=True)
        return tmp

    # 5) Still missing in production → fail fast with a clear message
    raise RuntimeError(
        "SECRET_KEY is required in production. "
        "Set one of: SECRET_KEY (env), SECRET_KEY_FILE (path), or SECRET_KEY_SECRET_NAME (GSM)."
    )

SECRET_KEY = _resolve_secret_key()
print(f"DEBUG: SECRET_KEY resolver => {'[SET]' if SECRET_KEY else '[NOT SET]'}", file=sys.stderr)

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
REFRESH_TOKEN_EXPIRE_SECONDS = 14 * 24 * 60 * 60  # 14 days
FS_TS = SERVER_TIMESTAMP
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
@app.options("/{path:path}")
def cors_preflight(path: str):
    # LB will inject the CORS headers we configured.
    return Response(status_code=204)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
@app.get("/")
async def root_ok():
    return {"status": "ok"}
@app.get("/_ah/health")
async def gclb_health():
    return {"status": "ok"}

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.head("/health")
async def health_head():
    return Response(status_code=200)

@app.head("/_ah/health")
async def gclb_health_head():
    return Response(status_code=200)

# CORS: only enable in local/dev; prod relies on Google Load Balancer
ENABLE_APP_CORS = os.getenv("ENABLE_APP_CORS", "0")  # "1" to enable locally

if ENABLE_APP_CORS == "1":
    origins = [
        "http://localhost:3000",
        # add any other local dev origins here
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/google-auth")


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
    # --- now = datetime.utcnow()

    conversations = db.collection("conversations")

    if conv_id:
        conv_ref = conversations.document(conv_id)
        await conv_ref.set({"user_id": user_id, "updated_at": FS_TS}, merge=True)
        doc = await conv_ref.get()
        return conv_ref, (doc.to_dict() or {})

    if resume_last:
        try:
            # your preferred ordered query (may require index)
            last = None
            q = (db.collection('conversations')
                .where('user_id', '==', user_id)
                .order_by('updated_at', direction=firestore.Query.DESCENDING)
                .limit(1))
            async for d in q.stream():
                last = d
                break
            if last:
                return last.reference, (last.to_dict() or {})
        except Exception as e:
            print('[resume_last] ordered query failed; falling back:', e)
            # Fallback: scan a reasonable number without order_by and pick newest in Python
            last_id, last_ts = None, None
            q = (db.collection('conversations')
                .where('user_id', '==', user_id)
                .limit(200))
            async for d in q.stream():
                doc = d.to_dict() or {}
                ts = doc.get('updated_at') or doc.get('created_at')
                if (last_ts is None) or (ts and ts > last_ts):
                    last_ts = ts
                    last_id = d.id
            if last_id:
                ref = db.collection('conversations').document(last_id)
                snap = await ref.get()
                return ref, (snap.to_dict() or {})
        # otherwise create a fresh conversation

    conv_ref = conversations.document()
    await conv_ref.set(
        {
            "user_id": user_id,
            "created_at": FS_TS,
            "updated_at": FS_TS,
            "subscription_id": cfg.get("subscription_id"),
            "title": cfg.get("title") or "New conversation",
            "summary": "",
        }
    )
    doc = await conv_ref.get()
    return conv_ref, (doc.to_dict() or {})

async def save_message(
    conv_ref,
    *,
    role: str,
    sender: str,
    content: str,
    file_metadata: Optional[dict] = None,
    file_metadata_list: Optional[List[dict]] = None,
):
    # now = firestore.SERVER_TIMESTAMP
    data = {
        "role": role,
        "sender": sender,
        "content": content,
        "created_at": FS_TS,
    }

    # Back-compat: keep the old single-file field if present
    if file_metadata:
        data["file_metadata"] = file_metadata

    # Preferred: a normalized list of attachments
    if file_metadata_list:
        data["attachments"] = file_metadata_list

    # save the message
    await conv_ref.collection("messages").add(data)

    # update parent conversation
    await conv_ref.set({
        "updated_at": FS_TS,
        "message_count": Increment(1),
    }, merge=True)



# --- End Conversation persistence helpers ---

@app.on_event("startup")
async def startup_event():
    print("STARTUP EVENT: scheduling Firestore init in background...")

    async def _init_subscriptions():
        try:
            subscriptions_ref = db.collection('subscriptions')
            plans = {
                'Free':       {'monthly_limit': 5,   'price_id': 'free_price_id_placeholder'},
                'Starter':    {'monthly_limit': 25,  'price_id': 'starter_price_id_placeholder'},
                'Pro':        {'monthly_limit': 200, 'price_id': 'pro_price_id_placeholder'},
                'Enterprise': {'monthly_limit': None,'price_id': 'enterprise_price_id_placeholder'},
            }
            for name, data in plans.items():
                doc_ref = subscriptions_ref.document(name)
                doc = await doc_ref.get()
                if not doc.exists:
                    await doc_ref.set(data)
            print("STARTUP EVENT: Firestore init done.")
        except Exception as e:
            print("STARTUP EVENT: Firestore init failed:", e)

    # do not await; let the server become 'ready' immediately
    asyncio.create_task(_init_subscriptions())


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

def _ts_iso(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return v.isoformat()
    except Exception:
        # last resort: stringify (works for Firestore Timestamp too)
        try:
            return str(v)
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

from typing import Dict

@app.get("/api/conversations/by_token")
async def list_conversations_by_token(
    token: Optional[str] = None,
    limit: int = 100,
    authorization: Optional[str] = Header(default=None),
):
    # Prefer Authorization header (fresh after /api/refresh); fall back to ?token=
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif not token:
        raise HTTPException(status_code=401, detail="Missing token")
    user = await get_current_user(token=token)
    # --- AUTO-BACKFILL-ONCE: run per-user on first visit if enabled ---
    import os
    if os.environ.get("AUTO_BACKFILL_ONCE", "").lower() in ("1", "true", "yes"):
        user_doc = db.collection("users").document(user["id"])
        snap = await user_doc.get()
        udoc = snap.to_dict() or {}
        if not udoc.get("backfill_convos_done"):
            try:
                updated = await _auto_backfill_for_user(user["id"], user.get("email"), limit=2000)
                await user_doc.set({
                    "backfill_convos_done": True,
                    "backfill_convos_count": updated,
                    "backfill_convos_ts": firestore.SERVER_TIMESTAMP,
                }, merge=True)
                print(f"[auto-backfill] user={user['id']} updated={updated}")
            except Exception as e:
                print("[auto-backfill] failed:", e)
    # --- /AUTO-BACKFILL-ONCE ---
    base = db.collection("conversations")
    items_by_id: Dict[str, dict] = {}

    # query by user_id
    q1 = base.where("user_id", "==", user["id"]).limit(limit)
    async for d in q1.stream():
        c = d.to_dict() or {}
        items_by_id[d.id] = {
            "id": d.id,
            "title": c.get("title") or "New conversation",
            "updated_at": _ts_iso(c.get("updated_at") or c.get("created_at")),
            "message_count": c.get("message_count", 0),
            "summary": c.get("summary", ""),
        }

    # legacy/email key fallback
    if user.get("email"):
        q2 = base.where("owner_keys", "array_contains", user["email"].lower()).limit(limit)
        async for d in q2.stream():
            c = d.to_dict() or {}
            items_by_id[d.id] = {
                "id": d.id,
                "title": c.get("title") or "New conversation",
                "updated_at": _ts_iso(c.get("updated_at") or c.get("created_at")),
                "message_count": c.get("message_count", 0),
                "summary": c.get("summary", ""),
            }

    items = list(items_by_id.values())
    # sort newest first; fall back to empty string
    items.sort(key=lambda x: (x.get("updated_at") or ""), reverse=True)
    return {"items": items[:limit]}



@app.get("/api/conversations/{conv_id}/messages")
async def list_messages(conv_id: str, limit: int = 50, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    conv = await conv_ref.get()
    if (not conv.exists) or ((conv.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = []
    q = (conv_ref.collection("messages")
         .order_by("created_at", direction=firestore.Query.DESCENDING)  # ← was 'timestamp'
         .limit(limit))
    async for m in q.stream():
        d = m.to_dict() or {}
        msgs.append({
            "id": m.id,
            "role": d.get("role"),
            "sender": d.get("sender"),
            "content": d.get("content"),
            "timestamp": _ts_iso(d.get("created_at")),  # ← was d.get("timestamp")
            # preferred list, plus legacy single object for back-compat if present
            "attachments": d.get("attachments") or ([d["file_metadata"]] if d.get("file_metadata") else []),
            "file_metadata": d.get("file_metadata"),  # keep for legacy readers
        })

    msgs.reverse()
    return {"items": msgs}

@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: RenameBody, user=Depends(get_current_user)):
    conv_ref = db.collection("conversations").document(conv_id)
    snap = await conv_ref.get()
    if (not snap.exists) or ((snap.to_dict() or {}).get("user_id") != user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")
    await conv_ref.update({"title": body.title, "updated_at": FS_TS})
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
    q = conv_ref.collection("messages").order_by("created_at")  # ← was 'timestamp'
    async for m in q.stream():
        d = m.to_dict() or {}
        msgs.append({
            "role": d.get("role"),
            "sender": d.get("sender"),
            "content": d.get("content"),
            "timestamp": _ts_iso(d.get("created_at")),  # ← was 'timestamp'
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


# ---------- ADMIN BACKFILL (one-off) ----------
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

class BackfillRequest(BaseModel):
    user_id: str
    user_email: Optional[str] = None
    human_senders: Optional[List[str]] = None  # e.g., ["Sam_Simpson", "Sam Simpson"]
    limit: int = 500
    apply: bool = False
    recompute_count: bool = False
    verbose: bool = False

def _iso_any(v):
    # tolerant ISO/str helper for logging
    if v is None: return None
    if isinstance(v, str): return v
    try:
        return v.isoformat()
    except Exception:
        try:
            return str(v)
        except Exception:
            return None

async def _message_bounds_and_count(conv_ref) -> Tuple[Optional[object], Optional[object], Optional[int]]:
    """Return (first_created_at, last_created_at, count). Count may be None if you don't want to scan."""
    msgs = conv_ref.collection("messages")

    first = None
    last = None

    # Latest
    async for m in msgs.order_by("created_at", direction=firestore.Query.DESCENDING).limit(1).stream():
        d = m.to_dict() or {}
        last = d.get("created_at")

    # Earliest
    async for m in msgs.order_by("created_at").limit(1).stream():
        d = m.to_dict() or {}
        first = d.get("created_at")

    # Counting can be heavy; do it only if asked
    cnt = 0
    async for _ in msgs.select([]).stream():
        cnt += 1

    return first, last, cnt

@app.post("/admin/backfill_conversations")
async def admin_backfill_conversations(
    payload: BackfillRequest,
    admin_secret: Optional[str] = Header(default=None, alias="x-admin-secret"),
):
    # Simple admin guard
    import os
    required = os.environ.get("ADMIN_SECRET")
    if not required or admin_secret != required:
        raise HTTPException(status_code=403, detail="Forbidden")

    user_id = payload.user_id
    email_lc = (payload.user_email or "").lower()
    human_senders = list(dict.fromkeys(payload.human_senders or []))  # de-dupe
    limit = max(1, min(payload.limit, 5000))

    base = db.collection("conversations")
    candidate_refs: Dict[str, Any] = {}

    # a) by user_id
    q1 = base.where("user_id", "==", user_id).limit(limit)
    async for d in q1.stream():
        candidate_refs[d.id] = d.reference

    # b) by owner_keys contains email (legacy)
    if email_lc:
        q2 = base.where("owner_keys", "array_contains", email_lc).limit(limit)
        async for d in q2.stream():
            candidate_refs[d.id] = d.reference

    # c) optional: via messages collection-group (sender in human_senders)
    # splits into chunks of 30 values for 'in' operator
    if human_senders:
        chunk = 30
        for i in range(0, len(human_senders), chunk):
            vals = human_senders[i:i+chunk]
            qg = db.collection_group("messages").where("sender", "in", vals).limit(limit)
            async for m in qg.stream():
                # parent is .../conversations/{id}/messages/{mid}
                parent = m.reference.parent.parent
                if parent:
                    candidate_refs[parent.id] = parent

    if not candidate_refs:
        return {"updated": 0, "dry_run": not payload.apply, "items": [], "candidates": 0}

    results = []
    updated = 0

    for conv_id, ref in candidate_refs.items():
        snap = await ref.get()
        if not snap.exists:
            continue
        doc = snap.to_dict() or {}
        changes = {}

        # Ensure owner fields
        if doc.get("user_id") != user_id:
            changes["user_id"] = user_id

        current_keys = set((doc.get("owner_keys") or []))
        need_keys = set()
        if user_id and user_id not in current_keys:
            need_keys.add(user_id)
        if email_lc and email_lc not in {k.lower() for k in current_keys}:
            need_keys.add(email_lc)
        if need_keys:
            changes["owner_keys"] = sorted({*(k.lower() for k in current_keys), *need_keys})

        # Title baseline
        if not (doc.get("title") and str(doc.get("title")).strip()):
            changes["title"] = "New conversation"

        # Compute/fill timestamps and count if missing or requested
        created_at = doc.get("created_at")
        updated_at = doc.get("updated_at")
        msg_count = doc.get("message_count")

        if payload.recompute_count or created_at is None or updated_at is None or msg_count in (None, 0):
            first_ts, last_ts, cnt = await _message_bounds_and_count(ref)
            created_at = created_at or first_ts
            updated_at = updated_at or last_ts or created_at
            if payload.recompute_count or (msg_count in (None, 0) and cnt is not None):
                msg_count = cnt

        if doc.get("created_at") is None and created_at is not None:
            changes["created_at"] = created_at
        if doc.get("updated_at") is None and updated_at is not None:
            changes["updated_at"] = updated_at
        if msg_count is not None and doc.get("message_count") != msg_count:
            changes["message_count"] = msg_count

        # Record what we'd do
        results.append({
            "id": conv_id,
            "changes": {k: _iso_any(v) for k, v in changes.items()},
        })

        # Apply if requested
        if payload.apply and changes:
            await ref.set(changes, merge=True)
            updated += 1

    return {
        "updated": updated,
        "dry_run": not payload.apply,
        "items": results[:200],  # include first 200 for visibility
        "candidates": len(candidate_refs),
    }

async def _msg_bounds_and_count(conv_ref):
    """Return (first_created_at, last_created_at, count) with minimal reads."""
    msgs = conv_ref.collection("messages")
    first = last = None
    cnt = 0

    async for m in msgs.order_by("created_at", direction=firestore.Query.DESCENDING).limit(1).stream():
        last = (m.to_dict() or {}).get("created_at")

    async for m in msgs.order_by("created_at").limit(1).stream():
        first = (m.to_dict() or {}).get("created_at")

    # Lightweight count (IDs only). If datasets are huge, you can comment this out.
    async for _ in msgs.select([]).stream():
        cnt += 1

    return first, last, cnt


async def _auto_backfill_for_user(user_id: str, user_email: str | None, limit: int = 1000) -> int:
    """Ensure existing conversations for this user are listable (owner_keys/timestamps/count). Returns #updated."""
    email_lc = (user_email or "").lower()
    base = db.collection("conversations")
    refs = {}

    # a) by user_id
    async for d in base.where("user_id", "==", user_id).limit(limit).stream():
        refs[d.id] = d.reference

    # b) by owner_keys contains email
    if email_lc:
        async for d in base.where("owner_keys", "array_contains", email_lc).limit(limit).stream():
            refs[d.id] = d.reference

    updated = 0
    for conv_id, ref in refs.items():
        snap = await ref.get()
        if not snap.exists:
            continue
        doc = snap.to_dict() or {}
        changes = {}

        # Owner fields
        if doc.get("user_id") != user_id:
            changes["user_id"] = user_id

        keys = set((doc.get("owner_keys") or []))
        need = set()
        if user_id and user_id not in keys:
            need.add(user_id)
        if email_lc and email_lc not in {k.lower() for k in keys}:
            need.add(email_lc)
        if need:
            changes["owner_keys"] = sorted({*(k.lower() for k in keys), *need})

        # Title baseline
        if not (doc.get("title") and str(doc.get("title")).strip()):
            changes["title"] = "New conversation"

        # Timestamps & count (fill if missing)
        created_at = doc.get("created_at")
        updated_at = doc.get("updated_at")
        msg_count = doc.get("message_count")

        if created_at is None or updated_at is None or msg_count in (None, 0):
            first, last, cnt = await _msg_bounds_and_count(ref)
            created_at = created_at or first
            updated_at = updated_at or last or created_at
            if msg_count in (None, 0) and cnt is not None:
                msg_count = cnt

        if doc.get("created_at") is None and created_at is not None:
            changes["created_at"] = created_at
        if doc.get("updated_at") is None and updated_at is not None:
            changes["updated_at"] = updated_at
        if msg_count is not None and doc.get("message_count") != msg_count:
            changes["message_count"] = msg_count

        if changes:
            await ref.set(changes, merge=True)
            updated += 1

    return updated

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
        # AFTER — environment-aware cookie
        # If you have the redirect URI here, this is an easy local/prod test:
        # --- BEFORE: environment-aware cookie ---       
        # Define a base cookie with production values by default.
        is_local = auth_code.redirect_uri.startswith("http://localhost") if "auth_code" in locals() else False

        # Define base cookie parameters with environment-specific values
        is_local = auth_code.redirect_uri.startswith("http://localhost") if "auth_code" in locals() else False

        cookie_kwargs = dict(
            key="refresh_token",
            value=refresh_token,
            max_age=REFRESH_TOKEN_EXPIRE_SECONDS,
            httponly=True,
            path="/",
            samesite="Lax",
            secure=not is_local,
            domain="localhost" if is_local else ".aicolosseum.app",
        )
        response.set_cookie(**cookie_kwargs)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_name": user_name,
            "user_id": user_id,
            "refresh_token": refresh_token, # <- NEW: return refresh token in body
        }

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")

# === STRIPE IMPLEMENTATION ===
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class SubscriptionRequest(BaseModel):
    price_id: str

from fastapi import Request, HTTPException

@app.post("/api/refresh")
async def refresh_access_token(request: Request):
    # 1) Get the refresh token from the HttpOnly cookie
    rt = request.cookies.get("refresh_token")
    if not rt:
        # Nothing to refresh with → 401
        raise HTTPException(status_code=401, detail="No refresh token")

    # 2) Validate & decode the refresh token
    try:
        payload = pyjwt.decode(rt, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # 3) Mint a new access token
    access_token = create_access_token(
        {"sub": sub},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"token": access_token, "access_token": access_token, "token_type": "bearer"}

@app.post("/api/logout")
async def logout(response: Response):
    # Delete possible old domain-scoped cookie
    response.delete_cookie(
        "refresh_token",
        path="/",
        domain=".aicolosseum.app",
        secure=True,
        samesite="Lax",
    )
    # Delete new host-only cookie
    response.delete_cookie(
        "refresh_token",
        path="/",
        secure=True,
        samesite="Lax",
    )
    return {"ok": True}


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
             .order_by("created_at", direction=firestore.Query.DESCENDING)  # ← was 'timestamp'
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
            "last_summary_at": FS_TS,
            "updated_at": FS_TS,
        })
    except Exception as e:
        print(f"[maybe_refresh_summary] skipped due to error: {e}")

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    

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

# --- File Upload Endpoint ---
async def _download_file(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    Receives a multipart file, uploads it to GCS, stores metadata in Firestore,
    and returns a short-lived signed URL plus basic metadata (JSON-safe).
    """
    try:
        # --- Required env vars ---
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not configured")

        # --- Filename & content-type ---
        filename = secure_filename((file.filename or "").strip())
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        content_type = file.content_type or (mimetypes.guess_type(filename)[0] or "application/octet-stream")

        # --- Build GCS object path ---
        file_path = f"uploads/{current_user['id']}/{datetime.now(timezone.utc).isoformat()}-{filename}"

        # --- Upload to GCS (off the event loop) ---
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        try:
            # Some UploadFile impls need this; won't hurt if not needed
            await file.seek(0)
        except Exception:
            try:
                file.file.seek(0)
            except Exception:
                pass

        await asyncio.to_thread(blob.upload_from_file, file.file, content_type=content_type)

        # --- Optional: size for UI ---
        size = None
        try:
            pos = file.file.tell()
            file.file.seek(0, os.SEEK_END)
            size = file.file.tell()
            file.file.seek(pos, os.SEEK_SET)
        except Exception:
            pass

        # --- Short-lived signed URL (IAM-based, Cloud Run friendly) ---
        service_account_email = os.getenv("GCP_SERVICE_ACCOUNT_EMAIL")
        if not service_account_email:
            raise HTTPException(status_code=500, detail="Missing GCP_SERVICE_ACCOUNT_EMAIL env var for IAM-based URL signing")

        adc_credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not adc_credentials.valid:
            adc_credentials.refresh(AuthRequest())

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET",
            service_account_email=service_account_email,
            access_token=adc_credentials.token,
        )

        # --- Small preview for text-like files (safe) ---
        content_for_llm = "[Binary file content not shown]"
        try:
            if content_type.startswith(("text/", "application/json", "application/xml", "application/javascript")):
                async with aiohttp.ClientSession() as session:
                    async with session.get(signed_url) as resp:
                        resp.raise_for_status()
                        content_for_llm = await resp.text()
                        if len(content_for_llm) > 10000:
                            content_for_llm = content_for_llm[:9500] + "\n.[TRUNCATED]"
        except Exception as e:
            print(f"Failed to read uploaded content: {e}")

        # --- Firestore: save an uploads doc so messages can refer to it by id ---
        doc_ref = db.collection("uploads").document()
        upload_doc = {
            "id": doc_ref.id,
            "user_id": current_user["id"],
            "name": filename,
            "mime": content_type,
            "size": size,
            "bucket": bucket_name,
            "path": file_path,
            "signed_url": signed_url,
            "content": content_for_llm,
            "created_at": FS_TS,   # Firestore server timestamp (Sentinel) -> DO NOT send to client
        }

        # IMPORTANT: await exactly once; do not call doc_ref.set(...) without await
        await doc_ref.set(upload_doc)  # this is AsyncDocumentReference.set(...)

        # --- JSON-safe response for the frontend ---
        response_doc = {
            "id": doc_ref.id,
            "name": filename,
            "mime": content_type,
            "size": size,
            "signed_url": signed_url,
            "content": content_for_llm,
            # optionally include a human timestamp if you want:
            # "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return JSONResponse(response_doc, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        print("upload_file error:", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/uploads/{upload_id}/text")
async def get_upload_text(upload_id: str, max_chars: int = 20000, current_user: dict = Depends(get_current_user)):
    """
    Return up to `max_chars` of plain text from a previously uploaded file.
    - For text-like mimetypes, read and return the first portion.
    - For binary types (PDF, images), try a quick text extraction (best-effort).
    """
    try:
        # 1) Look up the upload doc
        doc_ref = db.collection("uploads").document(upload_id)
        snapshot = await doc_ref.get()
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="Upload not found")

        data = snapshot.to_dict() or {}
        if data.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")

        bucket_name = data.get("bucket")
        path        = data.get("path")
        mime        = data.get("mime") or "application/octet-stream"
        if not bucket_name or not path:
            raise HTTPException(status_code=500, detail="Upload metadata incomplete")

        # 2) Fetch object from GCS (off the event loop)
        bucket = storage_client.bucket(bucket_name)
        blob   = bucket.blob(path)

        # Stream the first ~max_chars bytes; for text this is sufficient
        # (we read bytes then decode; you can refine for encodings if needed)
        def _download_head(n: int) -> bytes:
            return blob.download_as_bytes(start=0, end=n - 1)

        raw = await asyncio.to_thread(_download_head, max(4096, max_chars * 2))  # read a bit extra to be safe
        text = ""
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""

        # 3) If non-text and empty decode, do best-effort extraction (optional)
        if (not text.strip()) and (not mime.startswith("text/")):
            # Very light-weight heuristics; expand with pdf/docx parsers when you’re ready
            if mime in ("application/json", "application/xml", "application/javascript"):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = "[Binary file: no text extracted]"

        if len(text) > max_chars:
            text = text[:max_chars] + "\n.[TRUNCATED]"

        return JSONResponse({"upload_id": upload_id, "mime": mime, "text": text})
    except HTTPException:
        raise
    except Exception as e:
        print("get_upload_text error:", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)





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

        monthly_limit = user_subscription_data.get('monthly_limit')
        if monthly_limit is not None:
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
            if conversation_count >= monthly_limit:
                # Correct indentation starts here 👇
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
        # --- Ensure conversation doc is materialized & listable ---

        try:
            snap = await conv_ref.get()
            doc = snap.to_dict() or {}

            # server-side timestamps
            # now = firestore.SERVER_TIMESTAMP

            # stable owner keys (doc id; also email if you have it)
            owner_keys = [user["id"]]
            if user.get("email"):
                owner_keys.append(user["email"].lower())

            seed = {
                "user_id": user["id"],
                "subscription_id": user_data.get("subscription_id"),
                "owner_keys": ArrayUnion(owner_keys),
                "title": doc.get("title") or "New conversation",
                "created_at": doc.get("created_at") or FS_TS,
                "updated_at": FS_TS if not doc.get("updated_at") else doc.get("updated_at"),
                "message_count": doc.get("message_count", 0),
            }

            await conv_ref.set(seed, merge=True)

        except Exception as e:
            print("[ws] ensure conversation materialized failed:", e)

        # NEW: ensure owner_keys are present so listing can find legacy/alternate keys
        try:
            safe_keys = [user["id"]]
            if user.get("email"):
                safe_keys.append(user["email"].lower())
            await conv_ref.set({
                "user_id": user["id"],                      # keep single owner field
                "owner_keys": ArrayUnion(safe_keys)
            }, merge=True)
        except Exception as e:
            print("[ws] owner_keys upsert failed:", e)

        # Tell the client which conversation id we’re using
        await websocket.send_json({
            "sender": "System",
            "type": "conversation_id",
            "id": conv_ref.id,
        })
        
        # NEW: also send conversation meta so the sidebar can show it immediately
        try:
            snap = await conv_ref.get()
            doc = snap.to_dict() or {}
            await ws.send_json({
                "sender": "System",
                "type": "conversation_meta",
                "id": conv_ref.id,
                "title": (doc.get("title") or "New conversation"),
                "updated_at": _ts_iso(doc.get("updated_at") or doc.get("created_at")),
            })
        except Exception as e:
            print("[ws] conversation_meta send failed:", e)

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

        # === TOOL: get_upload_text (for assistants) ===
        async def get_upload_text_tool(upload_id: str, max_chars: int = 20000) -> dict:
            """
            Return up to `max_chars` of text from a user-uploaded file by its upload_id.
            Authorization: uses the current websocket user (no cross-user reads).
            """
            # 1) Look up the upload document and permission
            doc_ref = db.collection("uploads").document(upload_id)
            snapshot = await doc_ref.get()

            # If the caller passed a filename instead of the doc id, fall back to name lookup
            if not snapshot.exists:
                query = (db.collection("uploads")
                        .where("user_id", "==", user["id"])
                        .where("name", "==", upload_id)
                        .limit(1))
                async for doc in query.stream():
                    snapshot = doc
                    upload_id = doc.id  # normalize to the real id
                    break

            # Permission / existence check after fallback
            if (not snapshot.exists) or ((snapshot.to_dict() or {}).get("user_id") != user["id"]):
                return {"error": "not_found_or_forbidden"}


            data = snapshot.to_dict() or {}
            bucket_name = data.get("bucket")
            path        = data.get("path")
            mime        = data.get("mime") or "application/octet-stream"
            if not bucket_name or not path:
                return {"error": "metadata_incomplete"}

            # 2) Download a head slice from GCS (off the event loop)
            bucket = storage_client.bucket(bucket_name)
            blob   = bucket.blob(path)

            def _download_head(n: int) -> bytes:
                # Read roughly enough bytes to decode to ~max_chars safely
                return blob.download_as_bytes(start=0, end=max(1, n) - 1)

            raw = await asyncio.to_thread(_download_head, max(4096, max_chars * 2))
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = ""

            # 3) Light fallback for non-text types
            if (not text.strip()) and (not mime.startswith("text/")):
                if mime in ("application/json", "application/xml", "application/javascript"):
                    try:
                        text = raw.decode("utf-8", errors="replace")
                    except Exception:
                        text = ""
                if not text:
                    text = "[Binary file: no text extracted]"

            if len(text) > max_chars:
                text = text[:max_chars] + "\n.[TRUNCATED]"

            # Return a compact, model-friendly object
            return {"upload_id": upload_id, "mime": mime, "text": text}
        # === END TOOL ===


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
            now_mono = time.monotonic()
            last = RECENT_GREETS.get(key)

            if (not has_any) and (last is None or (now_mono - last) > GREETING_TTL_SECONDS):
                greeter = random.choice(agent_names)  # e.g. "ChatGPT", "Claude", "Gemini", "Mistral"
                greeting_text = f"Hi {user_display_name}, what can we help you with?"

                # Send greeting message immediately (no queue, so no risk of double-send)
                await websocket.send_json({"sender": greeter, "text": greeting_text})

                # Persist so your history shows the opener
                await save_message(conv_ref, role="assistant", sender=greeter, content=greeting_text)


                RECENT_GREETS[key] = now_mono

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
            
            # Tooling available to you:
            # - get_upload_text(upload_id: string, max_chars: integer) -> { "mime": string, "text": string }
            #   Use this whenever the user's provided `content` preview is insufficient.
            #   The `upload_id` is the `id` found in file attachments (file_metadata_list / attachments).
            #   Ask for only what you need (e.g., 10k–20k chars) to keep responses focused.

            return (
                f"{base}\n\n"
                f"Your name is {name}. Participants: {safe_user_name} (user), ChatGPT, Claude, Gemini, Mistral (assistants).\n"
                f"{base_prompt}"
                f"Ground your reply only in this chat. If you reference past statements, quote or paraphrase them from the visible messages. "
                f"If unsure, ask for a quick clarification instead of guessing.\n"
                f"Attribution: When asked to list who said what, use EXACT speaker names from the transcript "
                f"(ChatGPT, Claude, Gemini, Mistral, {safe_user_name}). "
                f"Do not merge, alias, or infer names."
            )


        # ---- model configs ----
        chatgpt_llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_type": "openai"
            }],
            "temperature": 0.5,
            "timeout": 90,
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

            async def a_generate_reply(self, messages=None, sender=None, **kwargs):
                print(f"[manager->assistant] speaker={self.name}")

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
                            # ---- Resolve OpenAI tool_calls inline to avoid dangling calls ----
                            try:
                                if isinstance(result, dict) and result.get("tool_calls"):
                                    tool_text_blocks = []
                                    for tc in result.get("tool_calls", []):
                                        fn = (tc.get("function") or {}).get("name")
                                        if fn == "get_upload_text":
                                            import json  # ok if already imported elsewhere
                                            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
                                            try:
                                                args = json.loads(args_raw)
                                            except Exception:
                                                args = {}
                                            u_id = args.get("upload_id")
                                            mchars = int(args.get("max_chars") or 20000)
                                            if u_id:
                                                # Execute the tool directly and turn it into plain text
                                                tool_res = await get_upload_text_tool(upload_id=u_id, max_chars=mchars)
                                                mime = (tool_res or {}).get("mime", "unknown")
                                                text = (tool_res or {}).get("text", "") or ""
                                                tool_text_blocks.append(f"[File {u_id} | {mime}]\n{text}")
                                    if tool_text_blocks:
                                        # Replace the assistant's tool call with normal content
                                        result = {"content": "\n\n".join(tool_text_blocks)}
                            except Exception:
                                # Degrade gracefully; never return a raw tool_calls message
                                result = {"content": "I tried to read the file but hit an internal error. Please try again."}
                            # ---- END resolve tool_calls ----

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

                    # DEBUG: log when the model produced nothing (helps confirm the root cause)
                    if not out_text:
                        print(f"[{self.name}] empty model output (result={repr(result)[:300]})")

                    # Guard: show a short message instead of pure silence
                    if not out_text:
                        out_text = "…(no content returned; please ask again or address me by name)"              

                    # Make sure the value we return up the stack is text, not a raw dict
                    result = out_text

                    # NEW: proactively forward the assistant's reply to the browser
                    if out_text:
                        try:
                            self._message_output_queue.put_nowait({
                                "sender": self.name,
                                "text": out_text
                            })
                        except Exception:
                            # best-effort fallback
                            await self._message_output_queue.put({
                                "sender": self.name,
                                "text": out_text
                            })


                    

                finally:
                    # ALWAYS clear typing even if we error/return early
                    try:
                        self._message_output_queue.put_nowait({"sender": self.name, "typing": False, "text": ""})
                    except Exception:
                        pass

                # Final safety: never return a non-string or empty payload
                if not isinstance(result, str):
                    try:
                        result = (result.get("content") or result.get("text") or "").strip()
                    except Exception:
                        result = ""

                if not result:
                    result = "…(no content returned; please ask again or address me by name)"

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

        # === Register tool for function-calling ===
        # Let each assistant *see* the tool definition (function schema)
        for a in agents[1:]:  # assistants only
            if a.name == "ChatGPT":
                continue
            a.register_for_llm(
                name="get_upload_text",
                description="Return up to max_chars of text from a user-uploaded file.",
            )(get_upload_text_tool)


        # Let the user proxy actually *execute* the tool call when the LLM requests it
        user_proxy.register_for_execution(
            name="get_upload_text"
        )(get_upload_text_tool)
        # === END Registration ===

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
                f"Participants: ChatGPT (assistant), Claude (assistant), Gemini (assistant), Mistral (assistant). "
                f"User: {safe_user_name} (human).\n"
                f"Speaker map (USE EXACT STRINGS): ChatGPT, Claude, Gemini, Mistral, {safe_user_name}.\n"
                f"Attribution rules:\n"
                f"• When listing who said what, copy the exact speaker names from the transcript; do not infer or rename.\n"
                f"• Never attribute any assistant's message to {safe_user_name}.\n"
                f"• If the user attached files, you may call get_upload_text(upload_id, max_chars) to read more from them when the preview is insufficient.\n"
                f"• If an assistant did not provide a number, say 'not provided' for that assistant only.\n"
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
                f"You are the group chat manager. Decide the single next speaker by exact name.\n\n"
                f"VALID SPEAKERS:\n"
                f"- {safe_user_name}  (the human user)\n"
                f"- ChatGPT\n- Claude\n- Gemini\n- Mistral\n\n"
                f"Rules:\n"
                f"1) If the user directly addresses an assistant by name, select that assistant.\n"
                f"2) If the user addresses *everyone* (phrases like 'everyone', 'all of you', 'each of you', "
                f"'all', 'y’all', 'you guys', 'all the models'), schedule answers from ChatGPT, Claude, Gemini, Mistral — "
                f"one per round (any sensible order). After all four respond, select {safe_user_name}.\n"
                f"   Do not skip due to redundancy — every assistant must reply once.\n"
                f"   Keep each reply concise; overlap is acceptable when asked for everyone.\n"
                f"3) If the user doesn’t specify anyone, prefer the last assistant who replied; otherwise choose the most relevant assistant.\n"
                f"4) When you want more input from the human, select {safe_user_name} (never 'User').\n"
                f"5) Use exact names only: {safe_user_name}, ChatGPT, Claude, Gemini, Mistral.\n\n"
                f"Output only one of those names and nothing else."
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
                    # Build a file list: prefer inline metadata; else resolve attachments by id
                    files = data.get("file_metadata_list") or []
                    if not files and data.get("file_metadata"):
                        files = [data["file_metadata"]]

                    # If only ids were sent, resolve them from Firestore
                    if not files:
                        att_ids = data.get("attachments") or []
                        resolved = []
                        for fid in att_ids:
                            if not fid:
                                continue
                            try:
                                snap = await db.collection("uploads").document(str(fid)).get()
                                if snap.exists:
                                    doc = snap.to_dict() or {}
                                    resolved.append({
                                        "id": snap.id,
                                        "filename": doc.get("name") or doc.get("filename") or "uploaded-file",
                                        "content_type": doc.get("mime") or doc.get("content_type") or "application/octet-stream",
                                        "size": doc.get("size"),
                                        "url": doc.get("signed_url") or doc.get("url"),
                                        "content": doc.get("content"),
                                    })
                            except Exception as e:
                                print(f"[ws] resolve attachment {fid} failed: {e}")
                        files = resolved

                    print(f"[ws] user message received: {len((data.get('message') or data.get('text') or ''))} chars, {len(files)} files")
                        
                    user_message = data.get("message", "") or data.get("text", "") or ""
                    # Allow files-only turns; ignore truly empty turns.
                    if not user_message:
                        file_count = len(files or [])
                        if file_count > 0:
                            user_message = f"(sent {file_count} file{'s' if file_count != 1 else ''})"
                        else:
                            # nothing to process; do not wake agents
                            continue

                    
                    # --- INSERT: allow files-only turns by synthesizing a placeholder ---
                    if not user_message:
                        file_count = len(files or [])
                        if file_count > 0:
                            user_message = f"(sent {file_count} file{'s' if file_count != 1 else ''})"
                        else:
                            # truly empty turn; ignore silently (or send a notice if you prefer)
                            continue
                    # --- /INSERT ---
                    # Build what the AIs should see: original text + per-file headers + previews
                    full_user_content = user_message
                    
                    # --- NEW: Initialize variables here to prevent crashes ---
                    image_urls = []
                    openai_parts = [{"type": "text", "text": full_user_content}]
                    
                    if files:
                        blocks = []
                        for f in files:
                            fn = (f.get("filename") or "uploaded-file").strip()
                            ct = (f.get("content_type") or "application/octet-stream").strip()
                            sz = f.get("size")
                            uid = f.get("id") or f.get("upload_id")
                            id_part = f" | upload_id={uid}" if uid else ""
                            header = f"[Attachment: {fn} | {ct} | {sz} bytes{id_part}]"
                            preview = f.get("content")
                            if isinstance(preview, str) and preview:
                                # trim to keep tokens in check
                                max_per_file = 9500
                                safe = preview[:max_per_file] + ("\n...[TRUNCATED]" if len(preview) > max_per_file else "")
                                blocks.append(header + "\n--- file preview start ---\n" + safe + "\n--- file preview end ---")
                            else:
                                url = f.get("url")
                                blocks.append(header + (f"\n(Downloadable URL: {url})" if url else ""))
                        full_user_content = (user_message + "\n\n" + "\n\n".join(blocks)).strip()
                        # Build OpenAI vision parts if any images are attached
                        image_urls = [
                            (f.get("url") or "")
                            for f in (files or [])
                            if str(f.get("content_type") or "").startswith("image/") and (f.get("url") or "")
                        ]
                        openai_parts = [{"type": "text", "text": full_user_content}]
                        for u in image_urls:
                            openai_parts.append({"type": "image_url", "image_url": {"url": u}})
                    
                    # ... rest of the code is here, which now safely uses image_urls and openai_parts
                    
                    # Correct indentation starts here 👇
                    # Save + echo the user turn exactly once (works for text-only and files)
                    await save_message(
                        conv_ref,
                        role="user",
                        sender=user_display_name,
                        content=user_message,
                        file_metadata_list=files,
                    )
                    await ws.send_json({
                        "sender": proxy.name,
                        "text": user_message,
                        "file_metadata_list": files,
                    })                                             
                    await maybe_set_title(conv_ref, user_message)

                    # NEW: push updated title/updated_at to the client
                    try:
                        snap = await conv_ref.get()
                        doc = snap.to_dict() or {}
                        await ws.send_json({
                            "sender": "System",
                            "type": "conversation_meta",
                            "id": conv_ref.id,
                            "title": (doc.get("title") or "New conversation"),
                            "updated_at": _ts_iso(doc.get("updated_at") or doc.get("created_at")),
                        })
                    except Exception as e:
                        print("[ws] conversation_meta (post-title) send failed:", e)

                    # If the user addressed everyone, give the manager a gentle, explicit hint
                    EVERYONE_PAT = re.compile(r"\b(everyone|all of you|each of you|all|y['’]all|you guys|all the models)\b", re.I)
                    if EVERYONE_PAT.search(user_message or ""):
                        try:
                            groupchat.messages.append({
                                "role": "system",
                                "content": (
                                    f"Manager directive: The user's last message addressed EVERYONE. "
                                    f"Schedule one concise reply from each assistant — ChatGPT, Claude, Gemini, and Mistral — even if content overlaps; "
                                    f"do not skip due to redundancy. After all four respond, select {safe_user_name}."
                                    f"If the user says only “thanks”, “thank you”, “ok”, or similar pleasantry, have the *last assistant who spoke* reply briefly."
                                    f"When the user addresses “everyone”, schedule one concise response per assistant. Do not suppress responses due to redundancy."
                                    f"Ground your reply only in this chat. If you reference earlier statements, quote or paraphrase them from the visible messages. If unsure, ask a brief clarification instead of guessing."

                                )

                            })
                        except Exception as _:
                            pass
                    # --- randomized opening greeting on brand-new conversations ---
                    try:
                        snap = await conv_ref.get()
                        doc = snap.to_dict() or {}
                        has_any = (doc.get("message_count") or 0) > 0
                    except Exception:
                        has_any = True

                    if not has_any:
                        opener = random.choice([a.name for a in agents if getattr(a, "name", "") in agent_names])
                        await ws.send_json({"type": "typing", "sender": opener, "typing": True})
                        await asyncio.sleep(random.uniform(0.4, 1.2))
                        await ws.send_json({
                            "sender": opener,
                            "text": random.choice([
                                f"Hi {user_display_name}, what can I help you with today?",
                                "Hello! Ready when you are.",
                            ])
                        })
                    # Kick off or feed the manager loop
                    if chat_task is None or chat_task.done():
                        # (Re)start the manager loop in the background

                    else:
                        # Feed subsequent user turns into the running loop
                        # Subsequent turns -> also text only.
                        await proxy.a_inject_user_message(full_user_content)




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
        # single, canonical version
        async def message_consumer_task(
            queue: asyncio.Queue,
            ws: WebSocket,
            conv_ref,
            agent_name_set: set,
            user_internal_name: str,
            user_display_name: str,
        ):
            last_text_by_sender: dict[str, str] = {}

            while True:
                msg = await queue.get()

                sender = (msg or {}).get("sender") or "System"
                text = (msg or {}).get("text") or (msg or {}).get("content") or ""
                is_typing_event = (msg and msg.get("typing") is not None)
    
                # Display-only cleanup: replace internal token with pretty name (handles markdown-escaped underscores)
                if text and sender in agent_name_set and user_internal_name:
                    # Build a pattern that treats '_' as possibly escaped '\_'
                    escaped = re.escape(user_internal_name).replace(r"\_", r"\\?_")
                    text = re.sub(escaped, user_display_name, text)


                # --- DEDUPE: skip duplicate non-empty texts from the same sender (typing still passes) ---
                if text and last_text_by_sender.get(sender) == text and not is_typing_event:
                    continue

                # Forward sanitized payload (and typing events) to the browser
                if text or is_typing_event:
                    payload = dict(msg)
                    payload["text"] = text  # ensure sanitized text goes out
                    # Guard against races / closed socket
                    if ws.application_state != WebSocketState.CONNECTED:
                        break
                    try:
                        await ws.send_json(payload)
                    except Exception as e:
                        print("[ws] send failed:", e)
                        break
                else:
                    # nothing to show
                    continue

                # Remember last text per sender after successful send (for future dedupe)
                if text:
                    last_text_by_sender[sender] = text

                # Persist only real assistant/system messages with content (skip user + typing-only)
                if sender != user_internal_name and text:
                    role = "assistant" if sender in agent_name_set else "system"
                    await save_message(conv_ref, role=role, sender=sender, content=text)
                    await maybe_refresh_summary(conv_ref)

        
        

        async def keepalive_task(ws: WebSocket):
            try:
                while True:
                    await asyncio.sleep(20)
                    if ws.application_state != WebSocketState.CONNECTED:
                        break
                    try:
                        await ws.send_json({"sender": "System", "type": "server_ping"})
                    except Exception:
                        break
            except Exception as e:
                print(f"keepalive_task error: {e}")
            
                
        # We start the tasks that are truly independent and long-running
        consumer_task_coro = asyncio.create_task(
            message_consumer_task(message_output_queue, websocket, conv_ref, agent_names, safe_user_name, user_display_name)
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

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("api:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8080")),
            lifespan="on",
            log_level="info")
