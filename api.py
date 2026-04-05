from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
from gemini_webapi import GeminiClient
from typing import Optional, List

app = FastAPI(title="Gemini Web API Wrapper")

# Load cookies from environment variables for security
SECURE_1PSID = os.getenv("SECURE_1PSID")
SECURE_1PSIDTS = os.getenv("SECURE_1PSIDTS")

client = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global client
    if not SECURE_1PSID:
        print("WARNING: SECURE_1PSID is not set in environment variables!")
        return
    
    client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
    await client.init()
    print("Gemini Client initialized successfully.")

@app.get("/")
async def root():
    return {"status": "running", "message": "Gemini Web API is ready"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized")
    
    try:
        if request.conversation_id:
            # Continue previous conversation if metadata/id is provided
            # Note: This is a simplified version. For real multi-turn, 
            # you'd need to manage session metadata.
            chat_session = client.start_chat() 
            # In a real app, you'd load metadata here
        else:
            chat_session = client.start_chat()
            
        response = await chat_session.send_message(request.message)
        return {
            "text": response.text,
            "conversation_id": chat_session.cid,
            "images": [img.url for img in response.images] if response.images else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
