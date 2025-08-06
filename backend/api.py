from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer


print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 2.0) <<")

load_dotenv()

app = FastAPI()

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

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    user_id: int

class GoogleIdToken(BaseModel):
    id_token: str

DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_size=20,
    max_overflow=40
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    monthly_limit = Column(Integer, nullable=True)
    users = relationship("User", back_populates="subscription")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), default=1)
    subscription = relationship("Subscription", back_populates="users")
    conversations = relationship("Conversation", back_populates="user")
    
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))  # Track conversation's subscription
    user = relationship("User", back_populates="conversations")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        free = db.query(Subscription).filter_by(name="Free").first()
        if not free:
            free = Subscription(name="Free", monthly_limit=5)
            db.add(free)
        
        starter = db.query(Subscription).filter_by(name="Starter").first()
        if not starter:
            starter = Subscription(name="Starter", monthly_limit=25)
            db.add(starter)

        pro = db.query(Subscription).filter_by(name="Pro").first()
        if not pro:
            pro = Subscription(name="Pro", monthly_limit=200)
            db.add(pro)
        
        enterprise = db.query(Subscription).filter_by(name="Enterprise").first()
        if not enterprise:
            enterprise = Subscription(name="Enterprise", monthly_limit=None)
            db.add(enterprise)
            
        db.commit()

        free_plan = db.query(Subscription).filter_by(name="Free").first()
        if free_plan:
            for user in db.query(User).filter(User.subscription_id == None).all():
                user.subscription_id = free_plan.id
            db.commit()
            
    except Exception as e:
        print(f"Error during startup: {e}")
        db.rollback()
    finally:
        db.close()

@app.get("/api/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "user_name": current_user.name,
        "user_id": current_user.id,
        "user_plan_name": current_user.subscription.name if current_user.subscription else "Free"
    }
    
@app.get("/api/users/me/usage")
def get_user_usage(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    first_day_of_month = date.today().replace(day=1)
    
    # Updated query to filter by subscription ID
    monthly_usage = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.timestamp >= first_day_of_month,
        Conversation.subscription_id == current_user.subscription.id
    ).count()
    
    monthly_limit = current_user.subscription.monthly_limit if current_user.subscription else 0
    
    return {
        "monthly_usage": monthly_usage,
        "monthly_limit": monthly_limit
    }

@app.post("/api/google-auth", response_model=Token)
async def google_auth(id_token: GoogleIdToken, db: Session = Depends(get_db)):
    try:
        GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        if not GOOGLE_CLIENT_ID:
            raise ValueError("Missing Google Client ID")

        idinfo = verify_oauth2_token(id_token.id_token, google_requests.Request(), GOOGLE_CLIENT_ID)

        user = db.query(User).filter_by(google_id=idinfo['sub']).first()

        if not user:
            user = User(google_id=idinfo['sub'], name=idinfo['name'], email=idinfo['email'])
            free_plan = db.query(Subscription).filter_by(name="Free").first()
            if free_plan:
                user.subscription_id = free_plan.id
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"New user created: {user.name} on Free plan.")
        else:
            print(f"User found in database: {user.name}")

        token = create_access_token({"sub": str(user.id)}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer", "user_name": user.name, "user_id": user.id}

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")

# === STRIPE IMPLEMENTATION ===
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class SubscriptionRequest(BaseModel):
    price_id: str

@app.post("/api/create-checkout-session")
async def create_checkout_session(request: SubscriptionRequest, current_user: User = Depends(get_current_user)):
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
            customer_email=current_user.email,
        )
        return {"id": checkout_session.id, "url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, stripe_webhook_secret
        )
    except ValueError as e:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the checkout.session.completed event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session["customer_email"]
        price_id = session["line_items"]["data"][0]["price"]["id"]
        
        # Find the user and the new subscription plan
        user = db.query(User).filter(User.email == customer_email).first()
        new_subscription = db.query(Subscription).filter(Subscription.price_id == price_id).first()

        if user and new_subscription:
            # Update the user's subscription
            user.subscription_id = new_subscription.id
            db.commit()
            print(f"User {user.email} successfully subscribed to the {new_subscription.name} plan.")
    
    return {"status": "success"}

# === END STRIPE IMPLEMENTATION ===

@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    try:
        await websocket.accept()

        user = get_current_user(token=token, db=db)
        
        # === SUBSCRIPTION LOGIC: CHECK USAGE LIMIT ===
        user_subscription = user.subscription
        if user_subscription and user_subscription.monthly_limit is not None:
            first_day_of_month = date.today().replace(day=1)
            conversation_count = db.query(Conversation).filter(
                Conversation.user_id == user.id,
                Conversation.timestamp >= first_day_of_month,
                Conversation.subscription_id == user_subscription.id
            ).count()
            
            if conversation_count >= user_subscription.monthly_limit:
                await websocket.send_json({
                    "sender": "System",
                    "text": "Your monthly conversation limit has been reached. Please upgrade your plan to continue."
                })
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Limit reached")
                return
        
        # Log the start of a new conversation
        new_conversation = Conversation(
            user_id=user.id,
            subscription_id=user.subscription.id if user.subscription else None
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)

        initial_config = await websocket.receive_json()

        print("INITIAL CONFIG RECEIVED:", initial_config)

        message_output_queue = asyncio.Queue()
        user_name = initial_config.get('user_name', 'User')

        sanitized_user_name = user_name.replace(" ", "_")

        # === AGENT CONFIGS ===
        # Dynamically create system messages with the user's name and explicit persona instructions
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
