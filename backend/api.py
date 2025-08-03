from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import autogen
from typing import List, Dict, Any, Union
import httpx
from jose import JWTError, jwt
from datetime import datetime, timedelta
from google.auth.transport import requests as google_requests
from google.oauth2.id_token import verify_oauth2_token
import re
import asyncio

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

print(">> THE COLOSSEUM BACKEND IS RUNNING (LATEST VERSION 2.0) <<")

load_dotenv()

app = FastAPI()

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(',')
origins = [origin.strip() for origin in CORS_ORIGINS]
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
    users = relationship("User", back_populates="subscription")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), default=1)
    subscription = relationship("Subscription", back_populates="users")

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
            free = Subscription(name="Free")
            db.add(free)
        db.commit()
        for user in db.query(User).all():
            user.subscription_id = free.id
        db.commit() 
    except:
        db.rollback()
    finally:
        db.close()

@app.get("/api/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"user_name": current_user.name, "user_id": current_user.id}

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
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"New user created: {user.name}")
        else:
            print(f"User found in database: {user.name}")

        token = create_access_token({"sub": str(user.id)}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer", "user_name": user.name, "user_id": user.id}

    except Exception as e:
        print(f"Google auth failed: {e}")
        raise HTTPException(status_code=401, detail="Google authentication failed")


@app.websocket("/ws/colosseum-chat")
async def websocket_endpoint(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    try:
        await websocket.accept()

        user = get_current_user(token=token, db=db)
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

        async def message_consumer():
            while True:
                msg = await message_output_queue.get()
                await websocket.send_json(msg)

        consumer_task = asyncio.create_task(message_consumer())
        
        conversation_task = asyncio.create_task(user_proxy.a_initiate_chat(manager, message=initial_config['message']))
        
        async def user_input_handler():
            while True:
                try:
                    data = await websocket.receive_json()
                    user_message = data['message']
                    await user_proxy.a_inject_user_message(user_message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error handling user input: {e}")
                    await websocket.send_json({"sender": "System", "text": f"Error: {e}"})
        
        input_handler_task = asyncio.create_task(user_input_handler())
        
        await asyncio.gather(consumer_task, conversation_task, input_handler_task)

    except WebSocketDisconnect:
        print("WebSocket closed")
    except Exception as e:
        print(f"General exception: {e}")
        await websocket.send_json({"sender": "System", "text": f"Error: {e}"})