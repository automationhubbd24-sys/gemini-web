from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import os
import asyncio
import json
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict
from sqlalchemy import create_all_engines, Column, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="Gemini Web API Advanced Wrapper")

# --- Database Setup (PostgreSQL) ---
DATABASE_URL = os.getenv("DATABASE_URL") # Coolify usually provides this
engine = create_engine(DATABASE_URL) if DATABASE_URL else create_engine("sqlite:///./test.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(String, primary_key=True, index=True)
    metadata_json = Column(Text) # To store Gemini session metadata

if DATABASE_URL:
    Base.metadata.create_all(bind=engine)

# --- Gemini Client Setup ---
SECURE_1PSID = os.getenv("SECURE_1PSID")
SECURE_1PSIDTS = os.getenv("SECURE_1PSIDTS")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

client = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_user"
    system_prompt: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global client
    if not SECURE_1PSID:
        print("CRITICAL: SECURE_1PSID is not set!")
        return
    
    client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
    await client.init()
    print("Gemini Client initialized successfully.")

@app.get("/")
async def root():
    return {"status": "running", "message": "Gemini Advanced API is ready"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized")
    
    db = SessionLocal()
    try:
        # 1. Load existing session from DB
        stored_session = db.query(ChatSessionModel).filter(ChatSessionModel.session_id == request.session_id).first()
        metadata = None
        if stored_session:
            metadata = json.loads(stored_session.metadata_json)

        # 2. Start/Resume Gemini Chat
        # Using the system prompt if provided in request, else use default from ENV
        current_system_prompt = request.system_prompt or SYSTEM_PROMPT
        
        # Note: In this wrapper, we can prepend the system prompt to the first message 
        # or use Gems if supported. For simplicity, we prepend to keep context.
        prompt_to_send = request.message
        if not stored_session:
            prompt_to_send = f"[SYSTEM PROMPT: {current_system_prompt}]\n\nUser: {request.message}"

        chat_instance = client.start_chat(metadata=metadata)
        response = await chat_instance.send_message(prompt_to_send)

        # 3. Save/Update session in DB
        new_metadata = json.dumps(chat_instance.metadata)
        if stored_session:
            stored_session.metadata_json = new_metadata
        else:
            new_session = ChatSessionModel(session_id=request.session_id, metadata_json=new_metadata)
            db.add(new_session)
        
        db.commit()

        return {
            "text": response.text,
            "session_id": request.session_id,
            "conversation_id": chat_instance.cid,
            "images": [img.url for img in response.images] if response.images else []
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
