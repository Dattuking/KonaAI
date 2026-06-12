import os
import json
import asyncio
import uuid
import sqlite3
import random
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from openai import OpenAI
from tavily import AsyncTavilyClient

# --- 1. CONFIGURATION INTERFACE & FRAMEWORK STARTUP ---
app = FastAPI(
    title="KonaAI Multimodal Vision Engine",
    description="Unified API orchestrating SQLite bucket persistence, profile photo structures, clipboard ingestion, and Llama Vision RAG data streams.",
    version="2.6.7"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "GLOBAL_SYSTEM_MASTER_PRODUCTION_TOKEN_KEY_FRAME")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    print("CRITICAL WARNING: Infrastructure keys are missing.")
    openai_client = None
    tavily_async_client = None
else:
    openai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=OPENAI_API_KEY)
    tavily_async_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# --- 2. STORAGE SYSTEM LAYERS (SQLITE MODEL ARCHITECTURE) ---
DB_DIR = "/data"
DB_FILE = os.path.join(DB_DIR, "kona_production_vault.db")

def init_db_schema():
    if not os.path.exists(DB_DIR):
        try: os.makedirs(DB_DIR, exist_ok=True)
        except Exception:
            global DB_FILE
            DB_FILE = "kona_production_fallback.db"

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            role_status TEXT NOT NULL,
            avatar_b64 TEXT,
            language_preference TEXT DEFAULT 'English',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            title TEXT NOT NULL,
            history_json TEXT NOT NULL,
            FOREIGN KEY (email) REFERENCES users (email)
        )
    """)
    conn.commit()
    conn.close()

init_db_schema()
RECOVERY_OTP_DB = {}

# --- Pydantic Data Contract Validations ---
class UserOnboardSchema(BaseModel):
    email: EmailStr
    password: str
    name: str
    dob: str
    role_status: str

class ForgotPasswordPayload(BaseModel):
    email: EmailStr

class VerifyOtpPayload(BaseModel):
    email: EmailStr
    otp_code: str

class ResetPasswordPayload(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str

class AvatarPayloadSchema(BaseModel):
    avatar_data: str

class TokenPayload(BaseModel):
    access_token: str
    token_type: str
    email: str

class ChatMessage(BaseModel):
    role: str
    content: str
    attachment_type: Optional[str] = None  
    attachment_name: Optional[str] = None
    image_data_uri: Optional[str] = None  

class SearchPayload(BaseModel):
    chat_id: Optional[str] = None
    history: List[ChatMessage]

# --- 3. CRYPTOGRAPHIC SUITE UTILITIES ---
def generate_hashed_bytes(password: str) -> str: return pwd_context.hash(password)
def verify_hashed_bytes(plain_password: str, hashed_password: str) -> bool: return pwd_context.verify(plain_password, hashed_password)
def sign_access_session_token(data: dict) -> str:
    bundle = data.copy()
    bundle.update({"exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    return jwt.encode(bundle, JWT_SECRET_KEY, algorithm=ALGORITHM)

def verify_active_session(token: str = Depends(oauth2_scheme)) -> str:
    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return decoded.get("sub")
    except JWTError: raise HTTPException(status_code=401, detail="Session expired.")

# --- 4. MULTIMODAL HYBRID VISION STREAM PIPELINE ---
async def aggregate_multimodal_vision_stream(email: str, chat_id: str, history: List[ChatMessage]):
    latest_turn = history[-1]
    user_text_prompt = latest_turn.content
    context_stream_accumulator = ""
    citations_payload_tracker = []

    try:
        if user_text_prompt:
            crawl_response = await tavily_async_client.search(query=user_text_prompt, max_results=3)
            for idx, res in enumerate(crawl_response.get('results', [])):
                anchor_id = idx + 1
                context_stream_accumulator += f"[{anchor_id}] Link: {res['url']}\nSummary: {res['content']}\n\n"
                citations_payload_tracker.append({"id": anchor_id, "title": res.get('title', 'Web Document'), "url": res['url']})
    except Exception: pass

    # Clean multi-line syntax flushing
    yield f"data: {json.dumps({'type': 'metadata', 'sources': citations_payload_tracker, 'chat_id': chat_id})}\n\n"
    await asyncio.sleep(0.01)

    try:
        system_rules = (
            "You are KonaAI, an advanced enterprise multimodal vision platform. Carefully analyze provided image bytes "
            "alongside query text tokens. Break down layout blocks, diagrams, and texts inside files systematically.\n\n"
            "VISUAL DESIGN PROTOCOLS:\n"
            "1. Wrap all technical source code or programming instructions strictly inside language code-blocks (e.g. ```python ... ```).\n"
            "2. Separate distinct segments with headings (##) and layout sections with markers (---)."
        )

        messages_bundle = [{"role": "system", "content": system_rules}]
        
        for node in history:
            if node.role == "user":
                content_structures = [{"type": "text", "text": f"{node.content}\n\n[Context Indices]:\n{context_stream_accumulator}"}]
                if node.image_data_uri:
                    content_structures.append({
                        "type": "image_url",
                        "image_url": {"url": node.image_data_uri}
                    })
                messages_bundle.append({"role": "user", "content": content_structures})
            else:
                messages_bundle.append({"role": "assistant", "content": node.content})

        llm_stream = openai_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages_bundle,
            temperature=0.2,
            stream=True
        )

        streaming_response_tracker = ""
        for chunk in llm_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                streaming_response_tracker += token
                # Formulate explicitly clean string packet terminations
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        updated_history = history.copy()
        updated_history.append(ChatMessage(role="assistant", content=streaming_response_tracker))
        history_str = json.dumps([m.model_dump() for m in updated_history])
        room_title = user_text_prompt[:30] + "..." if user_text_prompt else "Vision Analysis Thread"

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO chats (chat_id, email, title, history_json) VALUES (?, ?, ?, ?)",
                       (chat_id, email, room_title, history_str))
        conn.commit()
        conn.close()

    except Exception as stream_fault:
        yield f"data: {json.dumps({'type': 'error', 'message': f'Vision Exception: {str(stream_fault)}'})}\n\n"

# --- 5. ENDPOINT SERVICE ROUTING TOPOLOGY ---

@app.post("/api/v1/auth/register", tags=["Security Infrastructure"])
async def onboard_system_user(payload: UserOnboardSchema):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE email = ?", (payload.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered, please sign in.")
    
    hashed_pass = generate_hashed_bytes(payload.password)
    cursor.execute("""
        INSERT INTO users (email, hashed_password, name, dob, role_status) VALUES (?, ?, ?, ?, ?)
    """, (payload.email, hashed_pass, payload.name, payload.dob, payload.role_status))
    conn.commit()
    conn.close()
    return {"message": "Onboarding completed successfully."}

@app.post("/api/v1/auth/login", response_model=TokenPayload, tags=["Security Infrastructure"])
async def authenticate_system_user(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE email = ?", (form_data.username,))
    row = cursor.fetchone()
    conn.close()
    if not row or not verify_hashed_bytes(form_data.password, row[0]):
        raise HTTPException(status_code=401, detail="Authorization rejected: Invalid credentials.")
    return {"access_token": sign_access_session_token(data={"sub": form_data.username}), "token_type": "bearer", "email": form_data.username}

@app.post("/api/v1/auth/upload-avatar", tags=["Profile photo Management"])
async def upload_user_avatar_b64(payload: AvatarPayloadSchema, user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET avatar_b64 = ? WHERE email = ?", (payload.avatar_data, user_identity))
    conn.commit()
    conn.close()
    return {"message": "Profile avatar context committed successfully."}

@app.get("/api/v1/auth/profile", tags=["Security Infrastructure"])
async def get_user_profile_node(user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT email, created_at, name, dob, role_status, avatar_b64 FROM users WHERE email = ?", (user_identity,))
    row = cursor.fetchone()
    conn.close()
    return {"email": row[0], "created_at": row[1], "name": row[2], "dob": row[3], "role_status": row[4], "avatar_b64": row[5]}

@app.post("/api/v1/auth/forgot-password", tags=["Password Recovery"])
async def initiate_recovery(payload: ForgotPasswordPayload):
    otp = f"{random.randint(100000, 999999)}"
    RECOVERY_OTP_DB[payload.email] = {"code": otp, "expires_at": datetime.utcnow() + timedelta(minutes=15)}
    return {"message": "Code generated.", "mock_debug_otp": otp}

@app.post("/api/v1/auth/verify-otp", tags=["Password Recovery"])
async def verify_otp(payload: VerifyOtpPayload):
    if RECOVERY_OTP_DB.get(payload.email, {}).get("code") != payload.otp_code: raise HTTPException(status_code=400, detail="Invalid token.")
    return {"message": "Verified."}

@app.post("/api/v1/auth/reset-password", tags=["Password Recovery"])
async def reset_pass(payload: ResetPasswordPayload):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET hashed_password = ? WHERE email = ?", (generate_hashed_bytes(payload.new_password), payload.email))
    conn.commit()
    conn.close()
    return {"message": "Password updated."}

@app.get("/api/v1/chats")
async def list_chats(user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, title FROM chats WHERE email = ?", (user_identity,))
    rows = cursor.fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1]} for r in rows]

@app.get("/api/v1/chats/{chat_id}")
async def get_chat(chat_id: str, user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, title, history_json FROM chats WHERE chat_id = ? AND email = ?", (chat_id, user_identity))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"chat_id": row[0], "title": row[1], "history": json.loads(row[2])}

@app.post("/api/v1/search")
async def run_search(payload: SearchPayload, user_identity: str = Depends(verify_active_session)):
    if not payload.history:
        raise HTTPException(status_code=400, detail="History payload context missing.")
    target_chat_id = payload.chat_id if payload.chat_id else str(uuid.uuid4())
    return StreamingResponse(aggregate_multimodal_vision_stream(user_identity, target_chat_id, payload.history), media_type="text/event-stream")

@app.get("/")
async def health_check(): return {"status": "online", "framework": "Multimodal Vision Node Connected"}
