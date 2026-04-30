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
import secrets
import json
from pathlib import Path
from gemini_webapi import GeminiClient
from typing import Optional, List, Dict, Any
from database import init_db, get_db, APIKey, Cookie, Log
from sqlalchemy.orm import Session

app = FastAPI(title="Evola Gemini AI Studio Proxy")

# Security
security = HTTPBearer()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "evola123")

# In-memory client cache
active_clients: Dict[str, tuple[GeminiClient, float]] = {}
sessions: Dict[str, list] = {}
client_lock = asyncio.Lock()
rotation_index = 0
cookie_cache: List[Cookie] = []
last_db_refresh = 0

async def warm_up_clients():
    """Background task to pre-initialize all alive cookies."""
    print("Starting background warming of Gemini clients...")
    while True:
        try:
            db = next(get_db())
            cookies = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").all()
            for cookie in cookies:
                if cookie.gmail not in active_clients:
                    try:
                        # Initializing as Standard Gemini Web Client
                        client = GeminiClient(cookie.secure_1psid, cookie.secure_1psidts, is_aistudio=False)
                        await client.init(timeout=30, auto_refresh=True)
                        async with client_lock:
                            active_clients[cookie.gmail] = (client, time.time())
                    except Exception: pass
            db.close()
        except Exception: pass
        await asyncio.sleep(600)

URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

# --- OpenAI Compatible Models ---
class OpenAIMessage(BaseModel):
    role: str
    content: Any
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    user: Optional[str] = "default_user"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

# --- Helpers ---
async def download_files_from_message(message: str, client: GeminiClient = None) -> List[str]:
    urls = URL_REGEX.findall(message)
    temp_files = []
    if not urls: return []
    for url in urls:
        try:
            session = client.client if client else None
            if not session:
                from curl_cffi.requests import AsyncSession
                session = AsyncSession()
            response = await session.get(url, timeout=30)
            if response.status_code == 200:
                ext = ".bin"
                ct = response.headers.get("content-type", "")
                if "image" in ct: ext = ".png"
                elif "pdf" in ct: ext = ".pdf"
                fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(fd, 'wb') as f: f.write(response.content)
                temp_files.append(temp_path)
        except: pass
    return temp_files

def format_tools_as_instruction(tools: List[Dict[str, Any]]) -> str:
    """Simulate native function calling behavior via advanced prompting."""
    if not tools: return ""
    instruction = "\n\n[SYSTEM_PROTOCOL: AGENT_MODE_ACTIVE]\n"
    instruction += "You are an advanced AI Agent. Use tools by outputting ONLY a JSON block:\n"
    instruction += "```json\n{\"tool_call\": {\"name\": \"fn_name\", \"arguments\": {...}}}\n```\n"
    instruction += "Available Tools:\n"
    for tool in tools:
        f = tool.get("function", {})
        instruction += f"- {f.get('name')}: {f.get('description')}. Schema: {json.dumps(f.get('parameters', {}))}\n"
    return instruction

async def extract_transcript(messages: List[OpenAIMessage]) -> tuple[str, str]:
    """Extract system instruction and rebuild conversation for AI Studio behavior."""
    sys_inst = "You are a helpful AI assistant."
    transcript = []
    for msg in messages:
        if msg.role == "system":
            sys_inst = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif msg.role == "user":
            transcript.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            transcript.append(f"Assistant: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    transcript.append(f"Tool Call: {tc['function']['name']}({tc['function']['arguments']})")
        elif msg.role == "tool":
            transcript.append(f"Tool Response: {msg.content}")
    return sys_inst, "\n".join(transcript)

async def get_next_client(db: Session) -> GeminiClient:
    global active_clients, rotation_index, cookie_cache, last_db_refresh
    curr = time.time()
    
    # Refresh cache if empty or expired
    if not cookie_cache or (curr - last_db_refresh) > 60:
        cookies = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").all()
        # Store as simple dictionaries to avoid DetachedInstanceError
        cookie_cache = [{
            "id": c.id,
            "gmail": c.gmail,
            "secure_1psid": c.secure_1psid,
            "secure_1psidts": c.secure_1psidts,
            "status": c.status
        } for c in cookies]
        last_db_refresh = curr
        
    if not cookie_cache:
        raise HTTPException(status_code=503, detail="No active cookies or all cookies are dead. Please update your cookies.")
    
    selected = cookie_cache[rotation_index % len(cookie_cache)]
    rotation_index += 1
    
    # Check if client is already in cache
    if selected["gmail"] in active_clients:
        return active_clients[selected["gmail"]][0]
    
    async with client_lock:
        try:
            # Initializing as Standard Gemini Web Client (Better compatibility with regular cookies)
            client = GeminiClient(selected["secure_1psid"], selected["secure_1psidts"], is_aistudio=False)
            await client.init(timeout=30, auto_refresh=True)
            active_clients[selected["gmail"]] = (client, time.time())
            return client
        except Exception as e:
            # Mark cookie as dead in database
            db.query(Cookie).filter(Cookie.id == selected["id"]).update({"status": "dead"})
            db.add(Log(event_type="error", message=f"Failed to initialize client for {selected['gmail']}: {str(e)}", gmail=selected["gmail"]))
            db.commit()
            
            # Remove from local cache list and cache map
            if selected["gmail"] in active_clients:
                del active_clients[selected["gmail"]]
            cookie_cache = [c for c in cookie_cache if c["id"] != selected["id"]]
            
            if not cookie_cache:
                raise HTTPException(status_code=503, detail="All available cookies have failed. Please provide new cookies.")
            
            # Try the next one
            return await get_next_client(db)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    key = db.query(APIKey).filter(APIKey.key == credentials.credentials, APIKey.is_active == True).first()
    if not key: raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# --- Main API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_next_client(db)
    system_instruction, full_history = await extract_transcript(request.messages)
    
    # Comprehensive AI Studio Native Mapping with Fallback
    model_map = {
        "fast": "gemini-2.0-flash-exp",
        "thinking": "gemini-2.0-flash-thinking-exp",
        "pro": "gemini-1.5-pro",
        "gemini-3-flash": "gemini-2.0-flash-exp",
        "gemini-3-pro": "gemini-1.5-pro",
        "gemini-2.0-flash-thinking": "gemini-2.0-flash-thinking-exp",
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-2.5-flash": "gemini-2.0-flash-exp"
    }
    target_model = model_map.get(request.model.lower(), "gemini-2.0-flash-exp")

    try:
        # Log request for debugging
        db.add(Log(event_type="info", message=f"Processing request for model: {request.model} mapping to {target_model}"))
        db.commit()

        chat = client.start_chat(model=target_model)
        
        # Enhanced Prompt for better RPC stability
        final_prompt = f"SYSTEM: {system_instruction}\n\nHISTORY:\n{full_history}\n\nUSER: {request.messages[-1].content}"
        
        # Slightly longer timeout but with better feedback
        response = await asyncio.wait_for(chat.send_message(final_prompt), timeout=90)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": None
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI Studio RPC timed out. Check your cookies or network.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RPC Error: {str(e)}")

# --- Dashboard ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        static_path = Path("static/index.html")
        if static_path.exists():
            with open(static_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Evola Dashboard not found. Please ensure static/index.html exists."
    except Exception as e:
        return f"Error loading dashboard: {str(e)}"

# --- Dashboard API ---
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def login(request: LoginRequest):
    if request.username == ADMIN_USERNAME and request.password == ADMIN_PASSWORD:
        return {"status": "success", "token": "admin-session-token"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    return {
        "total_cookies": db.query(Cookie).count(),
        "active_cookies": db.query(Cookie).filter(Cookie.status == "alive").count(),
        "total_keys": db.query(APIKey).count()
    }

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

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(warm_up_clients())

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "fast", "owned_by": "google"},
        {"id": "thinking", "owned_by": "google"},
        {"id": "pro", "owned_by": "google"},
        {"id": "gemini-2.0-flash-thinking", "owned_by": "google"},
        {"id": "gemini-2.0-flash", "owned_by": "google"},
        {"id": "gemini-1.5-pro", "owned_by": "google"},
        {"id": "gemini-1.5-flash", "owned_by": "google"}
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
