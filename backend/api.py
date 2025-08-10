import sys
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

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION - MERGED) <<")

load_dotenv()

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

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
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
        plans = {
            'Free': {'monthly_limit': 5, 'price_id': 'free_price_id_placeholder'},
            'Starter': {'monthly_limit': 25, 'price_id': 'starter_price_id_placeholder'},
            'Pro': {'monthly_limit': 200, 'price_id': 'pro_price_id_placeholder'},
            'Enterprise': {'monthly_limit': None, 'price_id': 'enterprise_price_id_placeholder'},
        }

        for name, data in plans.items():
            doc_ref = subscriptions_ref.document(name)
            doc = await doc_ref.get()
            if not doc.exists:
                await doc_ref.set(data)
                print(f"STARTUP EVENT: Created subscription plan: {name}")
        print("STARTUP EVENT: Firestore initialization complete.")
    except Exception as e:
        print(f"STARTUP EVENT: Failed to initialize Firestore collections: {e}")
        raise

@app.get("/api/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    subscription_doc = await db.collection('subscriptions').document(current_user['subscription_id']).get()
    return {
        "user_name": current_user['name'],
        "user_id": current_user['id'],
        "user_plan_name": subscription_doc.id
    }
    
@app.get("/api/users/me/usage")
async def get_user_usage(current_user: dict = Depends(get_current_user)):
    try:
        user_data = current_user
        subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
        subscription_data = subscription_doc.to_dict()

        if subscription_data['monthly_limit'] is None:
            return {"monthly_usage": 0, "monthly_limit": None}

        first_day_of_month = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_usage_query = db.collection('conversations').where(
            'user_id', '==', current_user['id']
        ).where(
            'subscription_id', '==', user_data['subscription_id']
        ).where(
            'timestamp', '>=', first_day_of_month
        )

        aggregation_query = monthly_usage_query.count()
        count_result = await aggregation_query.get()
        monthly_usage = count_result[0][0].value

        return {
            "monthly_usage": monthly_usage,
            "monthly_limit": subscription_data['monthly_limit']
        }
    except Exception as e:
        print(f"USAGE_ENDPOINT: Error getting user usage: {e}")
        raise HTTPException(status_code=500, detail="Error getting user usage")

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
        else:
            user_id = user_list[0].id

        token = create_access_token({"sub": user_id}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer", "user_name": idinfo['name'], "user_id": user_id}

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")

# Stripe Implementation
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class SubscriptionRequest(BaseModel):
    price_id: str

@app.post("/api/create-checkout-session")
async def create_checkout_session(request: SubscriptionRequest, current_user: dict = Depends(get_current_user)):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": request.price_id, "quantity": 1}],
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
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session["customer_email"]
        price_id = session["line_items"]["data"][0]["price"]["id"]
        
        users_ref = db.collection('users')
        user_doc_query = users_ref.where('email', '==', customer_email).limit(1).stream()
        user_docs = [doc async for doc in user_doc_query]
        
        subscriptions_ref = db.collection('subscriptions')
        new_subscription_doc_query = subscriptions_ref.where('price_id', '==', price_id).limit(1).stream()
        new_subscription_list = [doc async for doc in new_subscription_doc_query]

        if user_docs and new_subscription_list:
            user_ref = users_ref.document(user_docs[0].id)
            await user_ref.update({'subscription_id': new_subscription_list[0].id})
    
    return {"status": "success"}

@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        await websocket.accept()
        user = await get_current_user(token=token)
        
        # Subscription check
        user_subscription_doc = await db.collection('subscriptions').document(user['subscription_id']).get()
        user_subscription_data = user_subscription_doc.to_dict()

        if user_subscription_data['monthly_limit'] is not None:
            first_day_of_month = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            conversation_count_query = db.collection('conversations').where(
                'user_id', '==', user['id']
            ).where('subscription_id', '==', user['subscription_id']
            ).where('timestamp', '>=', first_day_of_month)
            
            agg_query = conversation_count_query.count()
            count_result = await agg_query.get()
            conversation_count = count_result[0][0].value

            if conversation_count >= user_subscription_data['monthly_limit']:
                await websocket.send_json({"sender": "System", "text": "Your monthly conversation limit has been reached."})
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        
        await db.collection('conversations').add({
            'user_id': user['id'],
            'timestamp': datetime.utcnow(),
            'subscription_id': user['subscription_id'],
        })

        initial_config = await websocket.receive_json()
        message_output_queue = asyncio.Queue()
        user_name = initial_config.get('user_name', 'User')
        sanitized_user_name = user_name.replace(" ", "_")

        CHATGPT_SYSTEM = f"""Your name is ChatGPT...""" # Truncated for brevity
        CLAUDE_SYSTEM = f"""Your name is Claude..."""
        GEMINI_SYSTEM = f"""Your name is Gemini..."""
        MISTRAL_SYSTEM = f"""Your name is Mistral..."""
        GROUPCHAT_SYSTEM_MESSAGE = f"""You are the GroupChatManager..."""

        chatgpt_llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]}
        claude_llm_config = {"config_list": [{"model": "claude-3-5-sonnet-20240620", "api_key": os.getenv("ANTHROPIC_API_KEY")}]}
        gemini_llm_config = {"config_list": [{"model": "gemini-1.5-pro", "api_key": os.getenv("GEMINI_API_KEY")}]}
        mistral_llm_config = {"config_list": [{"model": "mistral-large-latest", "api_key": os.getenv("MISTRAL_API_KEY")}]}
        
        class WebSocketAssistantAgent(autogen.AssistantAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._message_output_queue = message_output_queue
            
            async def a_send_typing_indicator(self, is_typing: bool):
                await self._message_output_queue.put({"sender": self.name, "typing": is_typing, "text": ""})

            async def a_generate_reply(self, messages: List[Dict[str, Any]] = None, sender: autogen.ConversableAgent = None, **kwargs) -> Union[str, Dict, None]:
                await self.a_send_typing_indicator(is_typing=True)
                try:
                    reply = await super().a_generate_reply(messages=messages, sender=sender, **kwargs)
                finally:
                    await self.a_send_typing_indicator(is_typing=False)
                return reply

        class CustomSpeakerSelector:
            # ... (implementation from old working file)
            def __init__(self, agents, user_name):
                self.assistant_agents = [agent for agent in agents if agent.name not in [user_name, "System"]]
                self.multi_agent_reply_state = {"is_active": False, "agents_to_reply": []}
                self.agent_by_name = {agent.name: agent for agent in agents}
                self.message_history = []
                self.loop_detected = False
                self.waiting_on_user_to_break_loop = False
                self.user_name = user_name

            def __call__(self, last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
                # ... (rest of the logic)
                return self.agent_by_name[self.user_name]

        class WebSocketUserProxyAgent(autogen.UserProxyAgent):
            def __init__(self, *args, message_output_queue: asyncio.Queue, **kwargs):
                super().__init__(*args, **kwargs)
                self._user_input_queue = asyncio.Queue()
                self._message_output_queue = message_output_queue

            async def a_receive(self, message, sender, **kwargs):
                # ... (implementation from old working file)
                pass

            async def a_generate_reply(self, **kwargs):
                return await self._user_input_queue.get()

            async def a_inject_user_message(self, message: str):
                await self._user_input_queue.put(message)

        user_proxy = WebSocketUserProxyAgent(
            name=sanitized_user_name,
            human_input_mode="NEVER",
            message_output_queue=message_output_queue,
            code_execution_config=False,
        )

        agents = [user_proxy]
        agents.append(WebSocketAssistantAgent("ChatGPT", llm_config=chatgpt_llm_config, system_message=CHATGPT_SYSTEM, message_output_queue=message_output_queue))
        # ... append other agents

        selector = CustomSpeakerSelector(agents, user_name=sanitized_user_name)
        groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=50, speaker_selection_method=selector)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False, system_message=GROUPCHAT_SYSTEM_MESSAGE)

        # ... (rest of websocket logic from old file)

    except WebSocketDisconnect:
        print("WebSocket closed")
    except Exception as e:
        print(f"General exception in websocket: {e}")
        await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
