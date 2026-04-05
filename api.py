from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict

app = FastAPI(title="Gemini Web API Memory Wrapper")

# Load cookies from environment variables for security
SECURE_1PSID = os.getenv("SECURE_1PSID")
SECURE_1PSIDTS = os.getenv("SECURE_1PSIDTS")

client = None
# Simple In-Memory Session Storage
sessions: Dict[str, any] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    system_prompt: Optional[str] = "You are a helpful assistant."

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
    return {"status": "running", "message": "Gemini Memory API is ready"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized")
    
    try:
        # Load metadata if session exists
        metadata = sessions.get(request.session_id)
        
        # Start chat with previous metadata (Memory)
        chat_session = client.start_chat(metadata=metadata)
        
        # If it's a new session, prepend system prompt
        final_message = request.message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {request.system_prompt}]\n\nUser: {request.message}"
            
        response = await chat_session.send_message(final_message)
        
        # Save updated metadata back to memory
        sessions[request.session_id] = chat_session.metadata
        
        return {
            "text": response.text,
            "session_id": request.session_id,
            "conversation_id": chat_session.cid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
