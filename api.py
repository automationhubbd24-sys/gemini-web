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

app = FastAPI(title="Evola Gemini Web API")

# Security
security = HTTPBearer()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "evola123")

# In-memory client cache (gmail -> (GeminiClient, last_used))
active_clients: Dict[str, tuple[GeminiClient, float]] = {}
sessions: Dict[str, list] = {}
client_lock = asyncio.Lock()
rotation_index = 0
cookie_cache: List[Cookie] = []
last_db_refresh = 0

async def warm_up_clients():
    """Background task to pre-initialize all alive cookies in the pool."""
    print("Starting background warming of Gemini clients...")
    while True:
        try:
            db = next(get_db())
            cookies = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").all()
            for cookie in cookies:
                if cookie.gmail not in active_clients:
                    try:
                        print(f"Pre-initializing client for {cookie.gmail} in background...")
                        client = GeminiClient(cookie.secure_1psid, cookie.secure_1psidts)
                        await client.init(timeout=30, auto_refresh=True)
                        async with client_lock:
                            active_clients[cookie.gmail] = (client, time.time())
                        print(f"Successfully warmed up {cookie.gmail}")
                    except Exception as e:
                        print(f"Background warming failed for {cookie.gmail}: {e}")
            db.close()
        except Exception as e:
            print(f"Warming loop error: {e}")
        
        await asyncio.sleep(600) # Re-check every 10 minutes

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
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    user: Optional[str] = "default_user"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

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
    """Extract system prompt, full conversation history and any file URLs from OpenAI messages."""
    system_prompt = "You are a helpful assistant."
    history = []
    file_urls = []
    
    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            elif isinstance(msg.content, list):
                system_prompt = " ".join([item.get("text", "") for item in msg.content if item.get("type") == "text"])
        
        elif msg.role == "user":
            text = ""
            if isinstance(msg.content, str):
                text = msg.content
                file_urls.extend(URL_REGEX.findall(msg.content))
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if item.get("type") == "text":
                        t = item.get("text", "")
                        text += t + " "
                        file_urls.extend(URL_REGEX.findall(t))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url: file_urls.append(url)
                    elif item.get("type") == "file_url":
                        url = item.get("file_url", {}).get("url", "")
                        if url: file_urls.append(url)
            history.append(f"User: {text.strip()}")
            
        elif msg.role == "assistant":
            if isinstance(msg.content, str):
                history.append(f"Assistant: {msg.content}")
            # Support for simulated tool calls in history
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    f = tc.get("function", {})
                    history.append(f"Assistant Tool Call: {f.get('name')}({f.get('arguments')})")
                    
        elif msg.role == "tool":
            history.append(f"[TOOL_RESULT (ID: {msg.tool_call_id}): {msg.content}]")
    
    temp_files = []
    unique_urls = list(dict.fromkeys(file_urls))
    for url in unique_urls:
        try:
            files = await download_files_from_message(url, client)
            temp_files.extend(files)
        except: pass
            
    return system_prompt, "\n".join(history).strip(), temp_files

def format_tools_instruction(tools: List[Dict[str, Any]]) -> str:
    """Format tools list into a system instruction for Gemini."""
    if not tools:
        return ""
    
    instruction = "\n\n[SYSTEM_NOTE: AGENT_MODE_ACTIVE]\n"
    instruction += "You are an AI Agent equipped with external tools. You MUST follow these rules strictly:\n"
    instruction += "1. If you need to use a tool to provide an accurate answer, you MUST NOT output a conversational response. Instead, you MUST output ONLY a JSON code block in the following format:\n"
    instruction += "```json\n{\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}, \"id\": \"call_abc123\"}}\n```\n"
    instruction += "2. DO NOT explain your reasoning before the JSON block.\n"
    instruction += "3. DO NOT output any text other than the JSON block when calling a tool.\n"
    instruction += "4. Use tools for any search, information gathering, or computation.\n\n"
    instruction += "Available Tools List:\n"
    for tool in tools:
        f = tool.get("function", {})
        instruction += f"- Name: {f.get('name')}\n"
        instruction += f"  Description: {f.get('description')}\n"
        instruction += f"  Parameters: {json.dumps(f.get('parameters', {}))}\n\n"
    
    return instruction

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Detect and parse tool calls from Gemini's text output."""
    tool_calls = []
    # Look for JSON blocks or raw JSON containing "tool_call"
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```|(\{.*?"tool_call".*?\})', re.DOTALL)
    matches = json_pattern.findall(text)
    
    for match in matches:
        json_str = match[0] or match[1]
        try:
            data = json.loads(json_str)
            if "tool_call" in data:
                tc = data["tool_call"]
                tool_calls.append({
                    "id": tc.get("id", f"call_{secrets.token_hex(4)}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": json.dumps(tc.get("arguments", {}))
                    }
                })
        except:
            continue
            
    return tool_calls

async def get_next_client(db: Session) -> GeminiClient:
    """Get the next Gemini client in a Round-Robin fashion (A -> B -> C)."""
    global active_clients, rotation_index, cookie_cache, last_db_refresh
    
    # 1. Refresh cookie cache from DB every 60 seconds to avoid constant DB calls
    current_time = time.time()
    if not cookie_cache or (current_time - last_db_refresh) > 60:
        cookie_cache = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").order_by(Cookie.id).all()
        last_db_refresh = current_time
        
    if not cookie_cache:
        raise HTTPException(status_code=503, detail="No active Gemini cookies available")
    
    # 2. Prefer already initialized clients to save time
    for _ in range(len(cookie_cache)):
        if rotation_index >= len(cookie_cache):
            rotation_index = 0
        
        selected_cookie = cookie_cache[rotation_index]
        if selected_cookie.gmail in active_clients:
            client, _ = active_clients[selected_cookie.gmail]
            rotation_index = (rotation_index + 1) % len(cookie_cache)
            return client
        
        rotation_index = (rotation_index + 1) % len(cookie_cache)

    # 3. If no cached client is found, pick the one at rotation_index and init it
    if rotation_index >= len(cookie_cache):
        rotation_index = 0
    selected_cookie = cookie_cache[rotation_index]
    rotation_index = (rotation_index + 1) % len(cookie_cache)

    async with client_lock:
        if selected_cookie.gmail in active_clients:
            client, _ = active_clients[selected_cookie.gmail]
            return client
            
        try:
            client = GeminiClient(selected_cookie.secure_1psid, selected_cookie.secure_1psidts)
            await client.init(timeout=20, auto_refresh=True)
            active_clients[selected_cookie.gmail] = (client, time.time())
            return client
        except Exception as e:
            db.query(Cookie).filter(Cookie.gmail == selected_cookie.gmail).update({"status": "dead"})
            db.add(Log(event_type="error", message=f"Failed to init client: {str(e)}", gmail=selected_cookie.gmail))
            db.commit()
            cookie_cache = [] # Invalidate cache
            return await get_next_client(db)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    key_entry = db.query(APIKey).filter(APIKey.key == credentials.credentials, APIKey.is_active == True).first()
    if not key_entry:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return credentials.credentials

# --- Routes ---
@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(warm_up_clients()) # Start background warming
    print("Evola Gemini System Started with Background Warming.")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Evola Dashboard not found. Please ensure static/index.html exists."

@app.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_next_client(db)
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
    models = [
        {"id": "gemini-3-flash-thinking", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-3-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-3-flash", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "salesmanchatbot-ultimate", "object": "model", "created": int(time.time()), "owned_by": "evola"}
    ]
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_next_client(db)
    system_prompt, full_history, temp_files = await extract_content_and_files(request.messages, client)
    
    # Map model to Gemini 3
    model_lower = request.model.lower()
    if "thinking" in model_lower:
        gemini_model = "gemini-3-flash-thinking"
    elif "pro" in model_lower:
        gemini_model = "gemini-3-pro"
    elif "flash" in model_lower:
        gemini_model = "gemini-3-flash"
    elif "3" in model_lower:
        gemini_model = "gemini-3-flash"
    else:
        # Default for tool use is thinking model as it's much better at it
        gemini_model = "gemini-3-flash-thinking" if request.tools else "unspecified"

    # Add tools instruction if tools are provided
    tools_instruction = format_tools_instruction(request.tools)
    
    try:
        session_id = request.user
        metadata = sessions.get(session_id)
        chat_session = client.start_chat(metadata=metadata, model=gemini_model)
        
        # If metadata exists, Gemini already has history. We only send the latest turn.
        # But if there are tool results, they must be sent.
        if metadata:
            # Get the last message's content
            last_msg = request.messages[-1]
            last_text = ""
            if last_msg.role == "user":
                if isinstance(last_msg.content, str): last_text = last_msg.content
                elif isinstance(last_msg.content, list): 
                    last_text = " ".join([item.get("text", "") for item in last_msg.content if item.get("type") == "text"])
            elif last_msg.role == "tool":
                last_text = f"[TOOL_RESULT (ID: {last_msg.tool_call_id}): {last_msg.content}]"
            
            final_message = last_text
            if tools_instruction:
                final_message = f"{last_text}\n\n[SYSTEM_REMINDER: {tools_instruction}]"
        else:
            # New session: send the full history to prime the memory
            final_message = f"[SYSTEM_PROMPT: {system_prompt}]\n\n{full_history}"
            if tools_instruction:
                final_message += tools_instruction

        response = await chat_session.send_message(final_message, files=temp_files if temp_files else None)
        sessions[session_id] = chat_session.metadata
        
        # Check for tool calls in the response
        tool_calls = parse_tool_calls(response.text)
        
        message_out = {"role": "assistant", "content": response.text}
        finish_reason = "stop"
        
        if tool_calls:
            message_out["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": message_out, "finish_reason": finish_reason}],
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
    client = await get_next_client(db)
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
