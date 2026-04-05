from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import asyncio
import time
import re
import tempfile
import shutil
import random
import io
from pathlib import Path
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict, Any
from database import init_db, get_db, APIKey, Cookie, Log
from sqlalchemy.orm import Session

app = FastAPI(title="Evola Gemini Web API")

# Security
security = HTTPBearer()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "evola123")

# In-memory client cache (gmail -> GeminiClient)
active_clients: Dict[str, GeminiClient] = {}
sessions: Dict[str, list] = {}
client_lock = asyncio.Lock()

# URL Regex to detect links
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."

class OpenAIMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    user: Optional[str] = "default_user"

class LoginRequest(BaseModel):
    username: str
    password: str

# --- Helpers ---
async def download_files_from_message(message: str, client: GeminiClient = None) -> List[str]:
    """Detect and download files from a message, returns list of local temp file paths."""
    urls = URL_REGEX.findall(message)
    temp_files = []
    
    if not urls:
        return []
        
    for url in urls:
        try:
            # We use GeminiClient's session if available, otherwise create a temporary one
            session = client.client if client else None
            if not session:
                from curl_cffi.requests import AsyncSession
                session = AsyncSession()
            
            response = await session.get(url, timeout=30)
            if response.status_code == 200:
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
                    url_ext = Path(url.split("?")[0]).suffix
                    if url_ext: ext = url_ext

                fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(fd, 'wb') as f:
                    f.write(response.content)
                temp_files.append(temp_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            
    return temp_files

async def extract_content_and_files(messages: List[OpenAIMessage], client: GeminiClient = None) -> tuple[str, str, List[str]]:
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
                file_urls.extend(URL_REGEX.findall(msg.content))
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        user_message += text + " "
                        file_urls.extend(URL_REGEX.findall(text))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url: file_urls.append(url)
                    elif item.get("type") == "file_url":
                        url = item.get("file_url", {}).get("url", "")
                        if url: file_urls.append(url)
    
    temp_files = []
    unique_urls = list(dict.fromkeys(file_urls))
    for url in unique_urls:
        try:
            files = await download_files_from_message(url, client)
            temp_files.extend(files)
        except: pass
            
    return system_prompt, user_message.strip(), temp_files

async def get_random_client(db: Session) -> GeminiClient:
    """Get a random initialized Gemini client from the active pool in DB."""
    global active_clients
    async with client_lock:
        cookies = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").all()
        if not cookies:
            raise HTTPException(status_code=503, detail="No active Gemini cookies available")
        
        selected_cookie = random.choice(cookies)
        if selected_cookie.gmail in active_clients:
            return active_clients[selected_cookie.gmail]
        
        try:
            client = GeminiClient(selected_cookie.secure_1psid, selected_cookie.secure_1psidts)
            await client.init()
            active_clients[selected_cookie.gmail] = client
            return client
        except Exception as e:
            selected_cookie.status = "dead"
            db.add(Log(event_type="error", message=f"Failed to init client: {str(e)}", gmail=selected_cookie.gmail))
            db.commit()
            return await get_random_client(db)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    key_entry = db.query(APIKey).filter(APIKey.key == credentials.credentials, APIKey.is_active == True).first()
    if not key_entry:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return credentials.credentials

# --- Routes ---
@app.on_event("startup")
async def startup_event():
    init_db()
    print("Evola Gemini System Started.")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Evola Dashboard not found. Please ensure static/index.html exists."

@app.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_random_client(db)
    temp_files = await download_files_from_message(request.message, client)
    try:
        metadata = sessions.get(request.session_id)
        chat_session = client.start_chat(metadata=metadata)
        final_message = request.message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {request.system_prompt}]\n\nUser: {request.message}"
        
        response = await chat_session.send_message(final_message, files=temp_files if temp_files else None)
        if request.session_id:
            sessions[request.session_id] = chat_session.metadata
            
        return {
            "text": response.text,
            "session_id": request.session_id,
            "conversation_id": chat_session.cid,
            "images": [img.url for img in response.images] if response.images else []
        }
    finally:
        for f in temp_files:
            try: os.remove(f)
            except: pass

@app.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    return {"object": "list", "data": [{"id": "salesmanchatbot-ultimate", "object": "model", "created": int(time.time()), "owned_by": "evola"}]}

@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_random_client(db)
    system_prompt, user_message, temp_files = await extract_content_and_files(request.messages, client)
    try:
        session_id = request.user
        metadata = sessions.get(session_id)
        chat_session = client.start_chat(metadata=metadata)
        final_message = user_message
        if not metadata:
            final_message = f"[SYSTEM INSTRUCTION: {system_prompt}]\n\nUser: {user_message}"
            
        response = await chat_session.send_message(final_message, files=temp_files if temp_files else None)
        sessions[session_id] = chat_session.metadata
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    finally:
        for f in temp_files:
            try: os.remove(f)
            except: pass

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    client = await get_random_client(db)
    temp_path = None
    try:
        ext = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        chat_session = client.start_chat()
        p = prompt if prompt else "Please transcribe this audio file."
        response = await chat_session.send_message(p, files=[temp_path])
        return response.text if response_format == "text" else {"text": response.text}
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

# --- Dashboard ---
@app.post("/api/login")
async def login(request: LoginRequest):
    if request.username == ADMIN_USERNAME and request.password == ADMIN_PASSWORD:
        return {"status": "success", "token": "admin-session-token"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    return {"total_cookies": db.query(Cookie).count(), "active_cookies": db.query(Cookie).filter(Cookie.status == "alive").count(), "total_keys": db.query(APIKey).count()}

@app.get("/api/cookies")
async def list_cookies(db: Session = Depends(get_db)):
    return db.query(Cookie).all()

@app.post("/api/cookies")
async def add_cookie(data: dict, db: Session = Depends(get_db)):
    db.add(Cookie(gmail=data["gmail"], secure_1psid=data["secure_1psid"], secure_1psidts=data.get("secure_1psidts")))
    db.commit()
    return {"status": "success"}

@app.delete("/api/cookies/{gmail}")
async def delete_cookie(gmail: str, db: Session = Depends(get_db)):
    db.query(Cookie).filter(Cookie.gmail == gmail).delete()
    db.commit()
    active_clients.pop(gmail, None)
    return {"status": "success"}

@app.get("/api/keys")
async def list_keys(db: Session = Depends(get_db)):
    return db.query(APIKey).all()

@app.post("/api/keys")
async def create_key(data: dict, db: Session = Depends(get_db)):
    import secrets
    k = APIKey(key=f"evola-{secrets.token_hex(16)}", label=data.get("label", "New Key"))
    db.add(k); db.commit()
    return {"status": "success", "key": k.key}

@app.delete("/api/keys/{key_id}")
async def delete_key(key_id: int, db: Session = Depends(get_db)):
    db.query(APIKey).filter(APIKey.id == key_id).delete()
    db.commit()
    return {"status": "success"}

@app.get("/api/logs")
async def get_logs(db: Session = Depends(get_db)):
    return db.query(Log).order_by(Log.created_at.desc()).limit(50).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
