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
from pinecone import Pinecone

# --- 1. CONFIGURATION INTERFACE & FRAMEWORK STARTUP ---
app = FastAPI(
    title="KonaAI Enterprise Production Engine",
    description="Unified API orchestrating SQLite persistence, extended user profiling metadata, and password recovery systems.",
    version="2.4.0"
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
    print("CRITICAL WARNING: Infrastructure backend cluster variables are unassigned.")
    openai_client = None
    tavily_async_client = None
    pc_vector_index = None
else:
    openai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=OPENAI_API_KEY)
    tavily_async_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pc_vector_index = pc.Index("kona-knowledge-base")
        except Exception:
            pc_vector_index = None
    else:
        pc_vector_index = None

# --- 2. STORAGE SYSTEM LAYERS (SQLITE MODEL ARCHITECTURE) ---
DB_DIR = "/data"
DB_FILE = os.path.join(DB_DIR, "kona_production_vault.db")

def init_db_schema():
    if not os.path.exists(DB_DIR):
        try:
            os.makedirs(DB_DIR, exist_ok=True)
        except Exception:
            global DB_FILE
            DB_FILE = "kona_production_fallback.db"

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Structural creation query for Extended Profile schemas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            role_status TEXT NOT NULL,
            language_preference TEXT DEFAULT 'English',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Safe dynamic column migrations fallback setup
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN name TEXT DEFAULT 'dattu'")
        cursor.execute("ALTER TABLE users ADD COLUMN dob TEXT DEFAULT '2004-01-01'")
        cursor.execute("ALTER TABLE users ADD COLUMN role_status TEXT DEFAULT 'Student'")
    except sqlite3.OperationalError:
        pass  # Structural parameters already appended perfectly

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

# In-memory transient database tracker window handling ephemeral 6-digit tokens
RECOVERY_OTP_DB = {}

# --- Pydantic Data Contract Validations ---
class UserOnboardSchema(BaseModel):
    email: EmailStr
    password: str
    name: str
    dob: str
    role_status: str
    language_preference: Optional[str] = "English"

class ForgotPasswordPayload(BaseModel):
    email: EmailStr

class VerifyOtpPayload(BaseModel):
    email: EmailStr
    otp_code: str

class ResetPasswordPayload(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str

class TokenPayload(BaseModel):
    access_token: str
    token_type: str
    email: str

class ChatMessage(BaseModel):
    role: str
    content: str

class SearchPayload(BaseModel):
    chat_id: Optional[str] = None
    history: List[ChatMessage]

# --- 3. CRYPTOGRAPHIC SUITE UTILITIES ---
def generate_hashed_bytes(password: str) -> str:
    return pwd_context.hash(password)

def verify_hashed_bytes(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def sign_access_session_token(data: dict) -> str:
    payload_bundle = data.copy()
    expiration_horizon = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload_bundle.update({"exp": expiration_horizon})
    return jwt.encode(payload_bundle, JWT_SECRET_KEY, algorithm=ALGORITHM)

def verify_active_session(token: str = Depends(oauth2_scheme)) -> str:
    try:
        decoded_bundle = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_identity: str = decoded_bundle.get("sub")
        if user_identity is None:
            raise HTTPException(status_code=401, detail="Session validation missing identity identifier descriptors.")
        return user_identity
    except JWTError:
        raise HTTPException(status_code=401, detail="Session token validation mismatch.")

# --- 4. HYBRID SEMANTIC CORE ENGINE (RAG) ---
async def parse_vector_database_context(prompt_string: str) -> str:
    if not pc_vector_index or not openai_client:
        return ""
    try:
        embed_response = openai_client.embeddings.create(input=[prompt_string], model="text-embedding-3-small")
        query_coordinates = embed_response.data[0].embedding
        matches_matrix = pc_vector_index.query(vector=query_coordinates, top_k=3, include_metadata=True)
        compiled_vectors_context = ""
        for match_item in matches_matrix.get('matches', []):
            if 'text' in match_item.get('metadata', {}):
                compiled_vectors_context += f"[Internal Verified Repository Document]: {match_item['metadata']['text']}\n"
        return compiled_vectors_context
    except Exception:
        return ""

async def aggregate_hybrid_rag_stream(email: str, chat_id: str, history: List[ChatMessage]):
    latest_user_prompt = history[-1].content
    context_stream_accumulator = ""
    citations_payload_tracker = []

    async def track_web_crawlers():
        nonlocal context_stream_accumulator
        try:
            crawl_response = await tavily_async_client.search(query=latest_user_prompt, max_results=3)
            for structural_idx, resource in enumerate(crawl_response.get('results', [])):
                anchor_id = structural_idx + 1
                context_stream_accumulator += f"[{anchor_id}] URL reference: {resource['url']}\nData Summary: {resource['content']}\n\n"
                citations_payload_tracker.append({
                    "id": anchor_id, "title": resource.get('title', 'Verified Web Matrix Index Document'), "url": resource['url']
                })
        except Exception:
            context_stream_accumulator += "Global internet crawling fabric link temporarily unresponsive.\n"

    async def track_semantic_indexes():
        nonlocal context_stream_accumulator
        retrieved_vector_blocks = await parse_vector_database_context(latest_user_prompt)
        if retrieved_vector_blocks:
            context_stream_accumulator += f"\n--- Verified Corporate Repository Vectors context Data ---\n{retrieved_vector_blocks}\n"

    await asyncio.gather(track_web_crawlers(), track_semantic_indexes())

    yield f"data: {json.dumps({'type': 'metadata', 'sources': citations_payload_tracker, 'chat_id': chat_id})}\n\n"
    await asyncio.sleep(0.01)

    try:
        system_rules = (
            "You are KonaAI, an elite enterprise hybrid RAG search platform. Synthesize deeply authoritative, "
            "factual, and comprehensive responses by blending provided web crawl telemetry with internal repository documents.\n\n"
            "LAYOUT & VISUAL DESIGN INSTRUCTIONS (MANDATORY):\n"
            "1. NEVER output a solid wall of text. Break your reasoning down systematically.\n"
            "2. Use clear Markdown Headings (##) to separate distinct concept categories.\n"
            "3. Use Horizontal Rules (---) to cleanly isolate major sections of your summary.\n"
            "4. Use structural bolding (**text**) on critical metrics, dates, models, and core keys to make the layout scannable.\n"
            "5. Use clean bullet points (*) to detail exhaustive capability lists and lists of features without omitting items.\n\n"
            "CRITICAL PROTOCOLS:\n"
            "1. Mirror the user's language script 1:1. If prompted in English, stay completely in English.\n"
            "2. Ground your reasoning strictly inside the provided context variables. If facts collide, prioritize internal vectors.\n"
            "3. Reference claims using brackets matching source indices [1], [2]."
        )

        messages_bundle = [{"role": "system", "content": f"{system_rules}\n\nIngested Hybrid context streams:\n{context_stream_accumulator}"}]
        for network_node in history:
            messages_bundle.append({"role": network_node.role, "content": network_node.content})

        llm_stream = openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=messages_bundle, temperature=0.15, stream=True
        )

        streaming_response_tracker = ""
        for text_frame_chunk in llm_stream:
            if text_frame_chunk.choices and text_frame_chunk.choices[0].delta.content:
                token_text = text_frame_chunk.choices[0].delta.content
                streaming_response_tracker += token_text
                yield f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"
                await asyncio.sleep(0.002)

        updated_history = history.copy()
        updated_history.append(ChatMessage(role="assistant", content=streaming_response_tracker))
        history_str = json.dumps([m.model_dump() for m in updated_history])
        room_title = latest_user_prompt[:30] + "..."

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
        if cursor.fetchone():
            cursor.execute("UPDATE chats SET history_json = ? WHERE chat_id = ?", (history_str, chat_id))
        else:
            cursor.execute("INSERT INTO chats (chat_id, email, title, history_json) VALUES (?, ?, ?, ?)",
                           (chat_id, email, room_title, history_str))
        conn.commit()
        conn.close()

    except Exception as stream_fault:
        yield f"data: {json.dumps({'type': 'error', 'message': f'RAG Exception: {str(stream_fault)}'})}\n\n"

# --- 5. ENDPOINT SERVICE ROUTING TOPOLOGY ---

@app.post("/api/v1/auth/register", tags=["Security Infrastructure"])
async def onboard_system_user(payload: UserOnboardSchema):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE email = ?", (payload.email,))
    if cursor.fetchone():
        conn.close()
        # Enforce exact descriptive string requirement for matching targets
        raise HTTPException(status_code=400, detail="Email already registered, please sign in.")
    
    hashed_pass = generate_hashed_bytes(payload.password)
    current_time_stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    cursor.execute("""
        INSERT INTO users (email, hashed_password, name, dob, role_status, language_preference, created_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (payload.email, hashed_pass, payload.name, payload.dob, payload.role_status, payload.language_preference, current_time_stamp))
    
    conn.commit()
    conn.close()
    return {"message": "Onboarding operations completed successfully."}

@app.post("/api/v1/auth/login", response_model=TokenPayload, tags=["Security Infrastructure"])
async def authenticate_system_user(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE email = ?", (form_data.username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not verify_hashed_bytes(form_data.password, row[0]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization rejected: invalid credentials.")
    
    signed_session_token = sign_access_session_token(data={"sub": form_data.username})
    return {"access_token": signed_session_token, "token_type": "bearer", "email": form_data.username}

# --- UNIFIED 6-DIGIT PASSWORD RECOVERY ROUTING LOGIC ---

@app.post("/api/v1/auth/forgot-password", tags=["Password Recovery Framework"])
async def initiate_password_recovery(payload: ForgotPasswordPayload):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE email = ?", (payload.email,))
    user_exists = cursor.fetchone()
    conn.close()
    
    if not user_exists:
        raise HTTPException(status_code=404, detail="No registered account found with this email handle.")
    
    # Generate unique 6-digit numeric combination token string
    otp_token_code = f"{random.randint(100000, 999999)}"
    RECOVERY_OTP_DB[payload.email] = {
        "code": otp_token_code,
        "expires_at": datetime.utcnow() + timedelta(minutes=15)
    }
    
    # Standard production mock log delivery console feedback string rule
    print(f"\n[SECURE RECOVERY SYSTEM] OTP Code for {payload.email} target handle node is: {otp_token_code}\n")
    return {"message": "A 6-digit recovery unique code has been generated.", "mock_debug_otp": otp_token_code}

@app.post("/api/v1/auth/verify-otp", tags=["Password Recovery Framework"])
async def verify_recovery_token(payload: VerifyOtpPayload):
    saved_otp_record = RECOVERY_OTP_DB.get(payload.email)
    if not saved_otp_record:
        raise HTTPException(status_code=400, detail="No password reset request initialized for this user account.")
    
    if datetime.utcnow() > saved_otp_record["expires_at"]:
        RECOVERY_OTP_DB.pop(payload.email, None)
        raise HTTPException(status_code=400, detail="The verification token has expired. Please request a new code.")
    
    if saved_otp_record["code"] != payload.otp_code:
        raise HTTPException(status_code=401, detail="Invalid verification code token validation signature match.")
        
    return {"message": "Verification successful. You may proceed to assign a new password cipher."}

@app.post("/api/v1/auth/reset-password", tags=["Password Recovery Framework"])
async def commit_new_password_cipher(payload: ResetPasswordPayload):
    saved_otp_record = RECOVERY_OTP_DB.get(payload.email)
    if not saved_otp_record or saved_otp_record["code"] != payload.otp_code:
        raise HTTPException(status_code=403, detail="Unauthorized action: token verification parameters failed.")
    
    new_hashed_bytes = generate_hashed_bytes(payload.new_password)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET hashed_password = ? WHERE email = ?", (new_hashed_bytes, payload.email))
    conn.commit()
    conn.close()
    
    RECOVERY_OTP_DB.pop(payload.email, None)
    return {"message": "Password cipher updated successfully in persistent schema tables instead of the old record."}

@app.get("/api/v1/auth/profile", tags=["Security Infrastructure"])
async def get_user_profile_node(user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT email, created_at, name, dob, role_status FROM users WHERE email = ?", (user_identity,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Identity profile context missing from schema.")
    return {
        "email": row[0],
        "created_at": row[1],
        "name": row[2],
        "dob": row[3],
        "role_status": row[4]
    }

@app.get("/api/v1/chats", tags=["Chat History Layer"])
async def get_user_chat_list(user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, title FROM chats WHERE email = ?", (user_identity,))
    rows = cursor.fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1]} for r in rows]

@app.get("/api/v1/chats/{chat_id}", tags=["Chat History Layer"])
async def get_single_chat_history(chat_id: str, user_identity: str = Depends(verify_active_session)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, title, history_json FROM chats WHERE chat_id = ? AND email = ?", (chat_id, user_identity))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Requested historic chat session node was not found.")
    return {"chat_id": row[0], "title": row[1], "history": json.loads(row[2])}

@app.post("/api/v1/search", tags=["RAG Analytics Core Engine"])
async def process_rag_analytics_stream(payload: SearchPayload, user_identity: str = Depends(verify_active_session)):
    target_chat_id = payload.chat_id if payload.chat_id else str(uuid.uuid4())
    return StreamingResponse(aggregate_hybrid_rag_stream(user_identity, target_chat_id, payload.history), media_type="text/event-stream")

@app.get("/", tags=["Health Diagnostics"])
async def monitoring_heartbeat():
    return {"status": "online", "framework": "KonaAI Engine v2.4.0 Online"}
