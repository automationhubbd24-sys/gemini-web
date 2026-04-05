from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
import time
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict, Any

app = FastAPI(title="Gemini Web API OpenAI Wrapper")

# Load cookies from environment variables for security
SECURE_1PSID = os.getenv("SECURE_1PSID")
SECURE_1PSIDTS = os.getenv("SECURE_1PSIDTS")

client = None
# In-memory session storage (session_id -> metadata)
sessions: Dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."

# --- OpenAI Compatible Models ---
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    user: Optional[str] = "default_user"

@app.on_event("startup")
async def startup_event():
    global client
    if not SECURE_1PSID:
        print("WARNING: SECURE_1PSID is not set!")
        return
    
    client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
    await client.init()
    print("Gemini Client initialized successfully.")

@app.get("/")
async def root():
    return {"status": "running", "message": "Gemini OpenAI-Compatible API is ready"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized")
    
    try:
        metadata = None
        if request.session_id and request.session_id in sessions:
            metadata = sessions[request.session_id]
        
        chat_session = client.start_chat(metadata=metadata)
        
        final_message = request.message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {request.system_prompt}]\n\nUser: {request.message}"
            
        response = await chat_session.send_message(final_message)
        
        if request.session_id:
            sessions[request.session_id] = chat_session.metadata
            
        return {
            "text": response.text,
            "session_id": request.session_id,
            "conversation_id": chat_session.cid,
            "images": [img.url for img in response.images] if response.images else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenAI Compatible Endpoint ---
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemini-web",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized")
    
    try:
        # Extract system prompt and last user message
        system_prompt = "You are a helpful assistant."
        user_message = ""
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        # Use 'user' field as session_id for persistence
        session_id = request.user
        metadata = sessions.get(session_id)
        
        chat_session = client.start_chat(metadata=metadata)
        
        final_message = user_message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {system_prompt}]\n\nUser: {user_message}"
            
        response = await chat_session.send_message(final_message)
        
        # Save metadata
        sessions[session_id] = chat_session.metadata
        
        # Format response in OpenAI style
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
