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

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 2.0 - FIRESTORE) <<")

load_dotenv()

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
                print(f"STARTUP EVENT: Created subscription plan: {name}")
            else:
                print(f"STARTUP EVENT: Plan '{name}' already exists.")
        print("STARTUP EVENT: Firestore initialization complete.")
    except Exception as e:
        print(f"STARTUP EVENT: Failed to initialize Firestore collections: {e}")
        raise

@app.get("/api/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    user_doc = await db.collection('users').document(current_user['id']).get()
    user_data = user_doc.to_dict()
    
    subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
    subscription_data = subscription_doc.to_dict()
    
    return {
        "user_name": user_data['name'],
        "user_id": user_data['id'],
        "user_plan_name": subscription_doc.id
    }
    
@app.get("/api/users/me/usage")
async def get_user_usage(current_user: dict = Depends(get_current_user)):
    print(f"USAGE_ENDPOINT: User {current_user['id']} requested usage.")
    try:
        user_doc = await db.collection('users').document(current_user['id']).get()
        user_data = user_doc.to_dict()
        print(f"USAGE_ENDPOINT: User data found for {current_user['id']}.")

        subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
        subscription_data = subscription_doc.to_dict()
        print(f"USAGE_ENDPOINT: Subscription data found for {user_data['subscription_id']}.")

        if subscription_data['monthly_limit'] is None:
            print("USAGE_ENDPOINT: User has unlimited plan. Returning 0 usage.")
            return {
                "monthly_usage": 0,
                "monthly_limit": None
            }

        first_day_of_month = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_usage_query = db.collection('conversations').where(
            'user_id', '==', current_user['id']
        ).where(
            'subscription_id', '==', user_data['subscription_id']
        ).where(
            'timestamp', '>=', first_day_of_month
        )
        print("USAGE_ENDPOINT: Query created.")

        monthly_usage_docs = await monthly_usage_query.get()
        monthly_usage = len(monthly_usage_docs)
        print(f"USAGE_ENDPOINT: Found {monthly_usage} conversations this month.")

        return {
            "monthly_usage": monthly_usage,
            "monthly_limit": subscription_data['monthly_limit']
        }
    except Exception as e:
        print(f"USAGE_ENDPOINT: Error getting user usage: {e}")
        raise HTTPException(status_code=500, detail="Error getting user usage")

from google.oauth2.flow import Flow

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
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

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
            print(f"User {customer_email} successfully subscribed to the {new_subscription_list[0].id} plan.")
    
    return {"status": "success"}

@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        await websocket.accept()

        user = await get_current_user(token=token)
        
        user_doc = await db.collection('users').document(user['id']).get()
        user_data = user_doc.to_dict()
        
        user_subscription_doc = await db.collection('subscriptions').document(user_data['subscription_id']).get()
        user_subscription_data = user_subscription_doc.to_dict()

        if user_subscription_data['monthly_limit'] is not None:
            first_day_of_month = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            conversation_count_query = db.collection('conversations').where(
                'user_id', '==', user['id']
            ).where(
                'subscription_id', '==', user_data['subscription_id']
            ).where(
                'timestamp', '>=', first_day_of_month
            )

            conversation_docs = await conversation_count_query.stream()
            conversation_count = len(list(conversation_docs))
            
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
        message_output_queue = asyncio.Queue()
        user_name = initial_config.get('user_name', 'User')
        sanitized_user_name = user_name.replace(" ", "_")

        CHATGPT_SYSTEM = f"""Your name is ChatGPT. You are a helpful AI assistant. You are in a group chat with a user named {user_name} and three other AIs: Claude, Gemini, and Mistral. Refer to yourself in the first person (I, me, my). Do not attempt to pass the turn to another agent. Your response should conclude with your assigned termination phrase. Pay close attention to the entire conversation history. When prompted as a group, you must provide a direct and helpful response to the user's prompt. Your goal is to work with your team to solve the user's request. Conclude with 'TERMINATE' when the task is complete."""
        CLAUDE_SYSTEM = f"""Your name is Claude. You are a helpful AI assistant. You are in a group chat with a user named {user_name} and three other AIs: ChatGPT, Gemini, and Mistral. Refer to yourself in the first person (I, me, my). Do not attempt to pass the turn to another agent. Your response should conclude with your assigned termination phrase. Pay close attention to the entire conversation history. When prompted as a group, you must provide a direct and helpful response to the user's prompt. Your goal is to work with your team to solve the user's request. Conclude with 'Task Completed.' at the end of your response."""
        GEMINI_SYSTEM = f"""Your name is Gemini. You are a helpful AI assistant. You are in a group chat with a user named {user_name} and three other AIs: ChatGPT, Claude, and Mistral. Refer to yourself in the first person (I, me, my). Do not attempt to pass the turn to another agent. Your response should conclude with your assigned termination phrase. Pay close attention to the entire conversation history. When prompted as a group, you must provide a direct and helpful response to the user's prompt. Your goal is to work with your team to solve the user's request. Conclude with 'Task Completed.' at the end of your response."""
        MISTRAL_SYSTEM = f"""Your name is Mistral. You are a helpful AI assistant. You are in a group chat with a user named {user_name}, and three other AIs: ChatGPT, Claude, and Gemini. Refer to yourself in the first person (I, me, my). Do not attempt to pass the turn to another agent. Your response should conclude with your assigned termination phrase. Pay close attention to the entire conversation history. When prompted as a group, you must provide a direct and helpful response to the user's prompt. Your goal is to work with your team to solve the user's request. Conclude with 'Task Completed.' at the end of your response."""
        GROUPCHAT_SYSTEM_MESSAGE = f"""
        You are the GroupChatManager. You are in a chat with a user named {user_name}, a ChatGPT assistant, a Claude assistant, a Gemini assistant, and a Mistral assistant.
        The available agents are ['User', 'ChatGPT', 'Claude', 'Gemini', 'Mistral']. Your goal is to manage the conversation and select the next speaker.
        """

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
                await self._message_output_queue.put({
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
                        self.agent_by_name[self.user_name]._message_output_queue.put({
                            "sender": "System",
                            "text": "⚠️ Loop detected. Waiting for your input to continue the conversation..."
                        })
                    )
                    return self.agent_by_name[self.user_name]
                
                if last_speaker.name in [a.name for a in self.assistant_agents]:
                    for agent in self.assistant_agents:
                        if f"{agent.name.lower()}, would you like" in last_content or f"{agent.name.lower()} would you like" in last_content:
                            print(f"Detected turn pass from {last_speaker.name} to {agent.name}.")
                            self.multi_agent_reply_state["is_active"] = False
                            return self.agent_by_name[agent.name]

                if last_speaker.name == self.user_name:
                    multi_agent_keywords = ["everyone", "all of you", "both of you", "each of you", "all agents", "all AIs", "you all"]
                    if any(word in last_content for word in multi_agent_keywords):
                        print("User message contains multi-agent keywords. Activating multi-agent reply state.")
                        self.multi_agent_reply_state["is_active"] = True
                        self.multi_agent_reply_state["agents_to_reply"] = [agent.name for agent in self.assistant_agents]
                        return self.agent_by_name[self.multi_agent_reply_state["agents_to_reply"].pop(0)]

                    for agent in self.assistant_agents:
                        if agent.name.lower() in last_content:
                            self.multi_agent_reply_state["is_active"] = False
                            return self.agent_by_name[agent.name]

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
                    await self._message_output_queue.put({"sender": sender_name, "text": cleaned})

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
        agents.append(WebSocketAssistantAgent("ChatGPT", llm_config=chatgpt_llm_config, system_message=CHATGPT_SYSTEM, message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Claude", llm_config=claude_llm_config, system_message=CLAUDE_SYSTEM, message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Gemini", llm_config=gemini_llm_config, system_message=GEMINI_SYSTEM, message_output_queue=message_output_queue))
        agents.append(WebSocketAssistantAgent("Mistral", llm_config=mistral_llm_config, system_message=MISTRAL_SYSTEM, message_output_queue=message_output_queue))


        print("AGENTS IN GROUPCHAT:")
        for a in agents:
            print(f" - {a.name}")
        if len(agents) < 2:
            await websocket.send_json({"sender": "System", "text": "No AIs available for your subscription."})
            return
        
        selector = CustomSpeakerSelector(agents, user_name=sanitized_user_name)

        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
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
                await ws.send_json(msg)

        async def user_input_handler_task(ws: WebSocket, proxy: WebSocketUserProxyAgent):
            while True:
                try:
                    data = await ws.receive_json()
                    user_message = data['message']
                    await proxy.a_inject_user_message(user_message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error handling user input: {e}")
                    await ws.send_json({"sender": "System", "text": f"Error: {e}"})

        consumer_task = asyncio.create_task(message_consumer_task(message_output_queue, websocket))
        conversation_task = asyncio.create_task(user_proxy.a_initiate_chat(manager, message=initial_config['message']))
        input_handler_task = asyncio.create_task(user_input_handler_task(websocket, user_proxy))

        try:
            done, pending = await asyncio.wait([consumer_task, conversation_task, input_handler_task], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    except WebSocketDisconnect:
        print("WebSocket closed")
    except Exception as e:
        print(f"General exception: {e}")
        try:
            await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
        except WebSocketDisconnect:
            pass
