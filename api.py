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
def format_tools_instruction(tools, user_question=""):
    instruction = "\n=== MANDATORY TOOL USAGE ===\n"
    instruction += "You MUST use one of the tools below to answer this question.\n"
    instruction += "Do NOT answer directly. Do NOT say you don't have information.\n"
    instruction += "You MUST respond with ONLY a JSON object to call the tool.\n\n"
    
    instruction += "RESPONSE FORMAT - respond with ONLY this JSON, nothing else:\n"
    instruction += '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {"param": "value"}}..]}\n\n'
    
    instruction += "RULES:\n"
    instruction += "- Your ENTIRE response must be valid JSON only\n"
    instruction += "- No markdown, no code blocks, no explanation\n"
    instruction += "- No text before or after the JSON\n\n"
    
    instruction += "Available tools:\n\n"
    
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {})
        
        instruction += f"Tool: {name}\n"
        instruction += f"Description: {desc}\n"
        
        if params.get("properties"):
            instruction += "Parameters:\n"
            required_params = params.get("required", [])
            for param_name, param_info in params["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                is_required = "required" if param_name in required_params else "optional"
                instruction += f"  - {param_name} ({param_type}, {is_required}): {param_desc}\n"
        instruction += "\n"
    
    instruction += "=== END OF TOOLS ===\n\n"
    
    first_tool = tools[0] if tools else {}
    first_func = first_tool.get("function", first_tool)
    first_name = first_func.get("name", "tool")
    
    instruction += f'EXAMPLE: If the user asks a question, respond with:\n'
    instruction += '{"tool_calls": [{"name": "' + first_name + '", "arguments": {"input": "the user question here"}}..]}\n\n'
    
    instruction += "Now respond with the JSON to call the appropriate tool:\n\n"
    return instruction

def parse_tool_calls(response_text):
    import uuid
    cleaned = response_text.strip()
    if "```" in cleaned:
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1).strip()
    
    json_candidates = [cleaned]
    json_match = re.search(r'\{[\s\S]*"tool_calls"[\s\S]*\}', cleaned)
    if json_match:
        json_candidates.append(json_match.group(0))
    
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                raw_calls = parsed["tool_calls"]
                if isinstance(raw_calls, list) and len(raw_calls) > 0:
                    formatted_calls = []
                    for call in raw_calls:
                        tool_name = call.get("name", "")
                        arguments = call.get("arguments", {})
                        if isinstance(arguments, dict):
                            arguments_str = json.dumps(arguments, ensure_ascii=False)
                        else:
                            arguments_str = str(arguments)
                        
                        formatted_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_str
                            }
                        })
                    return formatted_calls
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return None

async def extract_transcript(messages: List[OpenAIMessage], tools=None) -> str:
    parts = []
    system_parts = []
    has_tool_results = False
    user_question = ""
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", item.get("content", str(item))))
                else:
                    text_parts.append(str(item))
            content = "\n".join(text_parts)
        
        if role == "system":
            system_parts.append(content)
        elif role == "tool":
            has_tool_results = True
            tool_name = getattr(msg, "name", "tool")
            parts.append(f"[TOOL RESULT from '{tool_name}']:\n{content}")
        elif role == "assistant":
            assistant_content = content if content else ""
            if msg.tool_calls:
                tc_descriptions = []
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    tc_descriptions.append(f"Called '{func.get('name', '?')}' with: {func.get('arguments', '{}')}")
                assistant_content += "\n[Previous tool calls: " + "; ".join(tc_descriptions) + "]"
            if assistant_content.strip():
                parts.append(f"[Assistant]: {assistant_content}")
        elif role == "user":
            user_question = content
            parts.append(content)
            has_tool_results = False
    
    final = ""
    if system_parts:
        if tools and not has_tool_results:
            final += "=== YOUR ROLE ===\n"
            final += "\n\n".join(system_parts)
            final += "\n=== END OF ROLE ===\n\n"
        else:
            final += "=== SYSTEM INSTRUCTIONS (FOLLOW STRICTLY) ===\n"
            final += "\n\n".join(system_parts)
            final += "\n=== END OF INSTRUCTIONS ===\n\n"
    
    if tools and not has_tool_results:
        final += format_tools_instruction(tools, user_question)
    
    if has_tool_results:
        final += "=== CONTEXT FROM TOOLS ===\n"
        final += "The following information was retrieved by the tools you requested.\n"
        final += "Use ONLY this information to answer the user's question.\n\n"
    
    if parts:
        final += "\n".join(parts)
    
    if has_tool_results:
        final += "\n\n=== INSTRUCTION ===\n"
        final += "Now answer the user's question based ONLY on the tool results above.\n"
    
    return final

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
    
    # Enhanced prompt building with tool support from mse_ai_api
    final_prompt = await extract_transcript(request.messages, tools=request.tools)
    
    # Comprehensive AI Studio Native Mapping with Fallback
    model_map = {
        "fast": "gemini-3-flash",
        "thinking": "gemini-3-flash-thinking",
        "pro": "gemini-3-pro",
        "gemini-3-flash": "gemini-3-flash",
        "gemini-3-pro": "gemini-3-pro",
        "gemini-2.0-flash-thinking": "gemini-3-flash-thinking",
        "gemini-2.0-flash": "gemini-3-flash",
        "gemini-1.5-pro": "gemini-3-pro",
        "gemini-1.5-flash": "gemini-3-flash",
        "gemini-2.5-flash": "gemini-3-flash"
    }
    target_model = model_map.get(request.model.lower(), "gemini-3-flash")

    try:
        # Log request for debugging
        db.add(Log(event_type="info", message=f"Processing request for model: {request.model} mapping to {target_model}"))
        db.commit()

        chat = client.start_chat(model=target_model)
        
        # Slightly longer timeout but with better feedback
        response = await asyncio.wait_for(chat.send_message(final_prompt), timeout=90)
        response_text = response.text
        
        # Parse tool calls if tools were provided
        tool_calls = None
        if request.tools:
            tool_calls = parse_tool_calls(response_text)

        if tool_calls:
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        else:
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": None
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    except asyncio.TimeoutError:
        db.add(Log(event_type="error", message=f"TimeoutError: Request timed out for model {request.model}"))
        db.commit()
        raise HTTPException(status_code=504, detail="AI Studio RPC timed out. Check your cookies or network.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        db.add(Log(event_type="error", message=f"RPC Error for model {request.model}: {str(e)}\nDetails: {error_details}"))
        db.commit()
        print(f"ERROR: {error_details}") # Print to console/logs for easier debugging
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
