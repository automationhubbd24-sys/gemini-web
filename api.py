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
                        client = GeminiClient(cookie.secure_1psid, cookie.secure_1psidts)
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
    if not cookie_cache or (curr - last_db_refresh) > 60:
        cookie_cache = db.query(Cookie).filter(Cookie.is_active == True, Cookie.status == "alive").all()
        last_db_refresh = curr
    if not cookie_cache: raise HTTPException(status_code=503, detail="No active cookies")
    
    selected = cookie_cache[rotation_index % len(cookie_cache)]
    rotation_index += 1
    if selected.gmail in active_clients: return active_clients[selected.gmail][0]
    
    async with client_lock:
        client = GeminiClient(selected.secure_1psid, selected.secure_1psidts)
        await client.init(timeout=20, auto_refresh=True)
        active_clients[selected.gmail] = (client, time.time())
        return client

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    key = db.query(APIKey).filter(APIKey.key == credentials.credentials, APIKey.is_active == True).first()
    if not key: raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# --- Main API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, token: str = Depends(verify_token), db: Session = Depends(get_db)):
    client = await get_next_client(db)
    system_instruction, full_history = await extract_transcript(request.messages)
    tools_instruction = format_tools_as_instruction(request.tools)
    
    # Map AI Studio models to Gemini Web Models
    model_map = {
        "gemini-3-flash-thinking": "gemini-3-flash-thinking",
        "gemini-3-pro": "gemini-3-pro",
        "gemini-3-flash": "gemini-3-flash",
        "gemini-2.5-flash": "gemini-3-flash",
        "gemini-2.0-flash-thinking": "gemini-3-flash-thinking",
        "gemini-2.0-flash": "gemini-3-flash",
        "gemini-1.5-pro": "gemini-3-pro",
        "gemini-1.5-flash": "gemini-3-flash"
    }
    target_model = model_map.get(request.model.lower(), "gemini-3-flash")
    
    # Build AI Studio like Prompt
    final_prompt = f"[SYSTEM_INSTRUCTION]\n{system_instruction}\n\n"
    final_prompt += f"[CONVERSATION_HISTORY]\n{full_history}\n\n"
    if tools_instruction: final_prompt += tools_instruction
    final_prompt += "\nAssistant:"

    try:
        # Using RPC for speed (Under the hood)
        chat = client.start_chat(model=target_model)
        response = await chat.send_message(final_prompt)
        
        # Parse for simulated tool calls
        tool_calls = []
        json_match = re.search(r'```json\s*({.*?})\s*```', response.text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool_call" in data:
                    tc = data["tool_call"]
                    tool_calls.append({
                        "id": f"call_{secrets.token_hex(4)}",
                        "type": "function",
                        "function": {"name": tc.get("name"), "arguments": json.dumps(tc.get("arguments", {}))}
                    })
            except: pass

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
                    "tool_calls": tool_calls if tool_calls else None
                },
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(warm_up_clients())

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "gemini-3-pro", "owned_by": "google"},
        {"id": "gemini-3-flash", "owned_by": "google"},
        {"id": "gemini-3-flash-thinking", "owned_by": "google"},
        {"id": "gemini-2.5-flash", "owned_by": "google"},
        {"id": "gemini-2.0-flash-thinking", "owned_by": "google"},
        {"id": "gemini-2.0-flash", "owned_by": "google"},
        {"id": "gemini-1.5-pro", "owned_by": "google"},
        {"id": "gemini-1.5-flash", "owned_by": "google"}
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
