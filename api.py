from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
import time
import re
import tempfile
from pathlib import Path
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict, Any

app = FastAPI(title="Gemini Web API OpenAI Wrapper")

# Load cookies from environment variables for security
SECURE_1PSID = os.getenv("SECURE_1PSID")
SECURE_1PSIDTS = os.getenv("SECURE_1PSIDTS")

client = None
# In-memory session storage (session_id -> metadata)
sessions: Dict[str, list] = {}

# URL Regex to detect links
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

async def download_files_from_message(message: str) -> List[str]:
    """Detect and download files from a message, returns list of local temp file paths."""
    urls = URL_REGEX.findall(message)
    temp_files = []
    
    if not urls:
        return []
        
    for url in urls:
        try:
            # We use GeminiClient's session if available, otherwise create a temporary one
            session = client.client if client and client.client else None
            if not session:
                from curl_cffi.requests import AsyncSession
                session = AsyncSession()
            
            response = await session.get(url, timeout=30)
            if response.status_code == 200:
                # Guess extension from content-type or URL
                content_type = response.headers.get("content-type", "")
                ext = ".bin"
                if "image/png" in content_type: ext = ".png"
                elif "image/jpeg" in content_type: ext = ".jpg"
                elif "image/webp" in content_type: ext = ".webp"
                elif "audio/mpeg" in content_type or "audio/mp3" in content_type: ext = ".mp3"
                elif "audio/ogg" in content_type: ext = ".ogg"
                elif "audio/wav" in content_type: ext = ".wav"
                elif "video/mp4" in content_type: ext = ".mp4"
                elif "application/pdf" in content_type: ext = ".pdf"
                else:
                    # Fallback to URL extension
                    url_ext = Path(url.split("?")[0]).suffix
                    if url_ext: ext = url_ext

                fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(fd, 'wb') as f:
                    f.write(response.content)
                temp_files.append(temp_path)
                print(f"Downloaded: {url} -> {temp_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            
    return temp_files

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."

# --- OpenAI Compatible Models ---
class OpenAIMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    user: Optional[str] = "default_user"

async def extract_content_and_files(messages: List[OpenAIMessage]) -> tuple[str, List[str], str]:
    """Extract system prompt, user message text and any file URLs from OpenAI messages."""
    system_prompt = "You are a helpful assistant."
    user_message = ""
    file_urls = []
    
    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            elif isinstance(msg.content, list):
                system_prompt = " ".join([item.get("text", "") for item in msg.content if item.get("type") == "text"])
        
        elif msg.role == "user":
            if isinstance(msg.content, str):
                user_message = msg.content
                # Detect URLs in text
                file_urls.extend(URL_REGEX.findall(msg.content))
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if item.get("type") == "text":
                        user_message += item.get("text", "") + " "
                        # Also detect URLs in the text inside list
                        file_urls.extend(URL_REGEX.findall(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            file_urls.append(url)
                    elif item.get("type") == "file_url": # Custom support for other files
                        url = item.get("file_url", {}).get("url", "")
                        if url:
                            file_urls.append(url)
    
    # Download all detected URLs
    temp_files = []
    # Use a set to avoid downloading the same URL twice
    unique_urls = list(dict.fromkeys(file_urls))
    
    for url in unique_urls:
        # Skip data URLs if they are huge (already handled by download_files_from_message if updated)
        # But for now let's use our download helper
        try:
            # Re-use our download logic but for a single URL
            files = await download_files_from_message(url) # This handles a single URL too
            temp_files.extend(files)
        except:
            pass
            
    return system_prompt, user_message.strip(), temp_files

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
    
    temp_files = []
    try:
        # Detect and download files
        temp_files = await download_files_from_message(request.message)
        
        metadata = None
        if request.session_id and request.session_id in sessions:
            metadata = sessions[request.session_id]
        
        chat_session = client.start_chat(metadata=metadata)
        
        final_message = request.message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {request.system_prompt}]\n\nUser: {request.message}"
            
        # Send message with files
        response = await chat_session.send_message(final_message, files=temp_files if temp_files else None)
        
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
    finally:
        # Cleanup temp files
        for f in temp_files:
            try: os.remove(f)
            except: pass

# --- OpenAI Compatible Endpoint ---
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "salesmanchatbot-ultimate",
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
    
    temp_files = []
    try:
        # Extract system prompt, user message and download any files/URLs
        system_prompt, user_message, temp_files = await extract_content_and_files(request.messages)
        
        # Use 'user' field as session_id for persistence
        session_id = request.user
        metadata = sessions.get(session_id)
        
        chat_session = client.start_chat(metadata=metadata)
        
        final_message = user_message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {system_prompt}]\n\nUser: {user_message}"
            
        # Send message with files
        response = await chat_session.send_message(final_message, files=temp_files if temp_files else None)
        
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
    finally:
        # Cleanup temp files
        for f in temp_files:
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
