import os
import re
import time
import json
from decimal import Decimal
from datetime import datetime, timedelta, timezone, date
from typing import Optional, List, Any, Dict
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from jose import jwt, JWTError
from passlib.context import CryptContext

from google import genai
from google.genai import types as genai_types
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "500"))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
ALLOW_FAKE_LOGIN = os.getenv("ALLOW_FAKE_LOGIN", "false").strip().lower() == "true"

CONTEXT_MESSAGES = 4
SUMMARY_PREVIEW_ROWS = 10
MAX_OUTPUT_TOKENS = 1024
SCHEMA_CACHE_TTL_SECONDS = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "300"))

if not SUPABASE_DB_URL:
    raise ValueError("SUPABASE_DB_URL is missing in .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is missing in .env")
if not GOOGLE_CLIENT_ID:
    raise ValueError("GOOGLE_CLIENT_ID is missing in .env")


# ============================================================
# DATABASE
# ============================================================
engine = create_engine(
    SUPABASE_DB_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# GEMINI CLIENT
# ============================================================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# PASSWORD HELPERS
# ============================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ============================================================
# JWT HELPERS
# ============================================================
def create_access_token(user_id: str, extra_claims: Optional[Dict[str, Any]] = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire, "iat": datetime.now(timezone.utc)}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    payload = decode_token_payload(token)
    return payload.get("sub") if payload else None


def decode_token_payload(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


# ============================================================
# AUTH DEPENDENCY
# ============================================================
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(401, "Not authenticated")

    user_id = decode_token(credentials.credentials)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")

    row = db.execute(
        text("SELECT user_id, name, email, profile_image_url, status, created_at "
             "FROM users WHERE user_id = :uid"),
        {"uid": user_id},
    ).mappings().first()

    if not row:
        raise HTTPException(401, "User not found")
    if row["status"] != "ACTIVE":
        raise HTTPException(403, "User account is not active")

    return dict(row)


# ============================================================
# GUEST MODE
# ============================================================
# Guests get real data + real SQL generation but nothing is persisted:
# no DB user row, no conversation/message rows. State lives entirely
# in-memory here, keyed by a per-login guest id ("gid"). This is
# intentionally process-local — a restart or a second worker resets
# guest limits, which is fine since guest sessions are meant to be
# throwaway.
GUEST_MAX_PROMPTS = 2
_guest_prompt_counts: Dict[str, int] = {}


def get_guest_actor(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(401, "Not authenticated")

    payload = decode_token_payload(credentials.credentials)
    if not payload or not payload.get("guest") or not payload.get("gid"):
        raise HTTPException(401, "Not a valid guest session")

    return {"gid": payload["gid"], "user_id": payload.get("sub")}


# ============================================================
# PYDANTIC MODELS
# ============================================================
class GoogleAuthRequest(BaseModel):
    id_token: str = Field(..., min_length=20)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    new_password: str = Field(..., min_length=8, max_length=128)


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default="New Conversation", max_length=255)


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    status: Optional[str] = Field(default=None)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = Field(default="en", max_length=5)


class GuestChatHistoryItem(BaseModel):
    role: str
    content: str = Field(default="", max_length=2000)


class GuestChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[GuestChatHistoryItem] = Field(default_factory=list)
    language: Optional[str] = Field(default="en", max_length=5)


# ============================================================
# APP + CORS
# ============================================================
app = FastAPI(
    title="DataQuery Agent",
    description="LLM-Powered Conversational Data Analytics and Query System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONSTANTS
# ============================================================
ANALYTICS_TABLES = {"products", "sales"}

SYSTEM_TABLES = {
    "users", "login_sessions", "conversations",
    "messages", "query_executions", "query_results",
}

FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXECUTE", "MERGE", "CALL",
    "VACUUM", "REINDEX", "CLUSTER", "COPY", "COMMENT",
]

# P0 #1 — greetings / noise filter
GREETING_WORDS = {
    "hi", "hie", "hey", "hello", "yo", "sup", "howdy",
    "thanks", "thank", "ok", "okay", "cool", "nice",
    "lol", "haha", "bye", "goodbye", "gm", "gn",
}

# P1 — hard cap on rows any query may return
SQL_ROW_LIMIT = 1000

REFUSE_MESSAGE = (
    "I can only answer questions about your sales data. "
    "Try something like: 'Show top 5 products by revenue' "
    "or 'What is total revenue by region?'"
)


# ============================================================
# P0 #1 — Python-side guard: is this a real question?
# ============================================================
def is_likely_data_question(text: str) -> bool:
    """
    Fast heuristic: return False for greetings / single words / nonsense
    so we never hit the LLM and never invent SQL for them.
    """
    q = text.strip().lower()

    # Empty or too short to be a real question
    if len(q) < 4:
        return False

    # Single word (e.g. "hi", "sales?")
    words = re.findall(r"[a-z0-9]+", q)
    if len(words) < 2:
        return False

    # All words are greetings/noise
    if all(w in GREETING_WORDS for w in words):
        return False

    # If the very first word is a greeting and nothing else, reject
    if words and words[0] in GREETING_WORDS and len(words) <= 2:
        return False

    return True


# ============================================================
# SQL VALIDATION
# ============================================================
def validate_sql(sql: str) -> tuple:
    if not sql or not sql.strip():
        return False, "Empty SQL"

    cleaned = sql.strip().rstrip(";").strip()

    if ";" in cleaned:
        return False, "Multiple SQL statements are not allowed"

    upper = cleaned.upper()

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Only SELECT or WITH queries are allowed"

    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"Forbidden keyword detected: {kw}"

    for tbl in SYSTEM_TABLES:
        if re.search(rf"\b{tbl}\b", cleaned, flags=re.IGNORECASE):
            return False, f"Access to system table '{tbl}' is not allowed"

    return True, "ok"


# ============================================================
# P1 — Auto-LIMIT expensive queries
# ============================================================
def enforce_row_limit(sql: str, limit: int = SQL_ROW_LIMIT) -> str:
    """
    If the SQL has no LIMIT clause, append one so a runaway query
    can't scan a huge table. Preserves any trailing semicolon.
    """
    cleaned = sql.strip()
    had_semicolon = cleaned.endswith(";")
    if had_semicolon:
        cleaned = cleaned[:-1].strip()

    if re.search(r"\bLIMIT\b", cleaned, flags=re.IGNORECASE):
        final = cleaned
    else:
        final = f"{cleaned} LIMIT {limit}"

    return final + (";" if had_semicolon else "")


# ============================================================
# SCHEMA INTROSPECTION (cached — schema rarely changes, and skipping
# the information_schema round-trips on every chat message is one of
# the biggest easy wins for response latency)
# ============================================================
_schema_cache: Dict[str, Any] = {"text": None, "ts": 0.0}


def get_analytics_schema(db: Session) -> str:
    now = time.time()
    if _schema_cache["text"] is not None and (now - _schema_cache["ts"]) < SCHEMA_CACHE_TTL_SECONDS:
        return _schema_cache["text"]

    schema_text = _load_analytics_schema(db)
    _schema_cache["text"] = schema_text
    _schema_cache["ts"] = now
    return schema_text


def _load_analytics_schema(db: Session) -> str:
    schema_lines: List[str] = []

    for table in sorted(ANALYTICS_TABLES):
        rows = db.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :t
                ORDER BY ordinal_position
            """),
            {"t": table},
        ).mappings().all()

        if not rows:
            continue

        cols = ", ".join(f"{r['column_name']}:{r['data_type']}" for r in rows)
        schema_lines.append(f"{table}({cols})")

    return "\n".join(schema_lines).strip()


# ============================================================
# CONVERSATION CONTEXT
# ============================================================
def load_recent_messages(db: Session, conversation_id: str, limit: int = CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    rows = db.execute(
        text("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = :cid
            ORDER BY created_at DESC
            LIMIT :lim OFFSET 1
        """),
        {"cid": conversation_id, "lim": limit},
    ).mappings().all()

    return [
        {"role": r["role"], "content": (r["content"] or "")[:300]}
        for r in reversed(rows)
    ]


# ============================================================
# GEMINI — retry + model fallback
# ============================================================
def call_gemini(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS, fast: bool = False) -> str:
    """
    fast=True puts the lightweight flash-lite model first — used for the
    natural-language summary, where speed matters more than raw reasoning
    power. SQL generation keeps the stronger flash model first for accuracy.
    """
    if fast:
        models_to_try = [
            "gemini-flash-lite-latest",
            GEMINI_MODEL,
            "gemini-3-flash-preview",
        ]
    else:
        models_to_try = [
            GEMINI_MODEL,
            "gemini-flash-lite-latest",
            "gemini-3-flash-preview",
        ]

    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    # thinking_budget=0 turns off "extended thinking" on Gemini 2.5/3 models —
    # for a short, structured task like SQL generation this is pure latency
    # with no quality benefit, so disabling it is the single biggest speed win.
    # Older models (e.g. gemini-flash-latest aliasing to a non-thinking model)
    # may reject the field, so we fall back to a plain config if that happens.
    fast_config = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    plain_config = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
    )

    last_error = None
    for model_name in models_to_try:
        config = fast_config
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                return (response.text or "").strip()
            except Exception as e:
                last_error = e
                err = str(e)
                if any(code in err for code in ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if config is fast_config and ("thinking" in err.lower() or "INVALID_ARGUMENT" in err):
                    # This model doesn't support thinking_config — retry once
                    # immediately without it before giving up on this model.
                    config = plain_config
                    continue
                raise
    raise last_error if last_error else RuntimeError("Gemini call failed")


def _map_llm_error(e: Exception) -> HTTPException:
    err = str(e)
    if "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
        return HTTPException(503, "The AI model is temporarily overloaded. Please try again in a few seconds.")
    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
        return HTTPException(429, "AI rate limit reached. Please wait a moment and try again.")
    if "404" in err or "NOT_FOUND" in err:
        return HTTPException(500, "AI model unavailable. Please contact support.")
    return HTTPException(500, "AI service error. Please try again.")


# ============================================================
# SQL GENERATION
# ============================================================
def generate_sql(question: str, schema: str, context: List[Dict[str, str]]) -> str:
    """
    P0 #2 — REFUSE is only allowed when the conversation has NO context.
    Follow-ups always attempt SQL generation.
    """
    context_block = ""
    if context:
        lines = [f"{m['role']}: {m['content']}" for m in context]
        context_block = (
            "Recent conversation (context only — do NOT repeat its SQL):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    has_context = len(context) > 0

    if has_context:
        # Follow-up path: never refuse, always generate SQL
        prompt = (
            "You are a PostgreSQL analyst for a sales dataset.\n\n"
            "The user is continuing an existing conversation. "
            "Write ONE SELECT query that answers the CURRENT QUESTION, "
            "using the recent conversation as context when useful.\n\n"
            "RULES:\n"
            "- Output ONLY the SQL. No markdown, no backticks, no explanation.\n"
            "- Use only tables/columns in the schema below.\n"
            "- Alias aggregates: AVG(x) AS avg_x, COUNT(*) AS cnt, SUM(x) AS total_x.\n"
            "- Add ORDER BY / LIMIT when useful.\n\n"
            f"SCHEMA:\n{schema}\n\n"
            f"{context_block}"
            f"CURRENT QUESTION: {question}\n\n"
            "SQL:"
        )
    else:
        # First message: allow REFUSE for non-data inputs
        prompt = (
            "You are a PostgreSQL analyst for a sales dataset.\n\n"
            "TASK: Read the CURRENT QUESTION below.\n"
            "- If it is a real question about the data, write ONE SELECT query that answers it.\n"
            "- If it is NOT a question about the data (greetings, small talk, random words, "
            "anything unrelated), reply with exactly the single word: REFUSE\n\n"
            "RULES when writing SQL:\n"
            "- Output ONLY the SQL. No markdown, no backticks, no explanation.\n"
            "- Use only tables/columns in the schema below.\n"
            "- Alias aggregates: AVG(x) AS avg_x, COUNT(*) AS cnt, SUM(x) AS total_x.\n"
            "- Add ORDER BY / LIMIT when useful.\n\n"
            f"SCHEMA:\n{schema}\n\n"
            f"CURRENT QUESTION: {question}\n\n"
            "ANSWER:"
        )

    sql = call_gemini(prompt, max_tokens=768)
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()
    return sql


def _normalize_value(v: Any) -> Any:
    """Coerce DB-native types (Decimal, date/datetime) to plain JSON/pandas-friendly types."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def execute_sql(db: Session, sql: str) -> tuple:
    start = time.time()
    result = db.execute(text(sql))
    rows = result.mappings().all()
    elapsed_ms = int((time.time() - start) * 1000)
    limited = [
        {k: _normalize_value(v) for k, v in dict(r).items()}
        for r in rows[:MAX_ROWS_RETURNED]
    ]
    return limited, elapsed_ms


# ============================================================
# CHART SELECTION
# ============================================================
def pick_chart(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    if df.empty or len(df.columns) < 2:
        return {"chart_type": "table", "config": None}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols or not other_cols:
        return {"chart_type": "table", "config": None}

    x_col = other_cols[0]
    y_col = numeric_cols[0]

    if len(numeric_cols) >= 2 and len(other_cols) == 0:
        return {
            "chart_type": "scatter",
            "config": {
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
            },
        }

    is_date = False
    try:
        pd.to_datetime(df[x_col])
        is_date = True
    except (ValueError, TypeError):
        is_date = False

    n_unique = df[x_col].nunique()

    if is_date:
        chart_type = "line"
    elif n_unique <= 6:
        chart_type = "pie"
    elif n_unique <= 30:
        chart_type = "bar"
    else:
        chart_type = "table"

    if chart_type == "table":
        return {"chart_type": "table", "config": None}

    return {
        "chart_type": chart_type,
        "config": {
            "x_axis": x_col,
            "y_axis": y_col,
            "title": f"{y_col} by {x_col}",
        },
    }


# ============================================================
# NATURAL LANGUAGE SUMMARY
# ============================================================
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}


def generate_summary(question: str, df: pd.DataFrame, language: str = "en") -> str:
    if df.empty:
        return "No data matched."

    preview = df.head(SUMMARY_PREVIEW_ROWS).to_string(index=False)
    lang_name = LANGUAGE_NAMES.get((language or "en").lower(), "English")
    lang_instruction = (
        f"Respond in {lang_name}. Keep numbers/currency in normal digits.\n"
        if lang_name != "English"
        else ""
    )

    prompt = (
        "Summarize the result in 1-2 short sentences with the key numbers.\n"
        "No SQL, no tables, no code.\n"
        f"{lang_instruction}\n"
        f"Q: {question}\n"
        f"DATA:\n{preview}\n"
        "A:"
    )

    try:
        text = call_gemini(prompt, max_tokens=200, fast=True)
        return text if text else f"{len(df)} rows returned."
    except Exception as e:
        print(f"[summary] gemini error: {e}")
        return f"{len(df)} rows returned."


# ============================================================
# ROOT / HEALTH
# ============================================================
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "DataQuery Agent is running"}


@app.get("/db-test", tags=["Health"])
def db_test(db: Session = Depends(get_db)):
    try:
        version = db.execute(text("SELECT version();")).scalar()
        tables = db.execute(
            text("SELECT table_name FROM information_schema.tables "
                 "WHERE table_schema='public' AND table_type='BASE TABLE' "
                 "ORDER BY table_name")
        ).scalars().all()
        return {"status": "connected", "version": version, "tables": list(tables)}
    except Exception as e:
        raise HTTPException(500, f"DB connection failed: {e}")


# ============================================================
# AUTH ENDPOINTS
# ============================================================
def _issue_login(db: Session, request: Request, user: Dict[str, Any], profile_image_url: Optional[str] = None) -> Dict[str, Any]:
    if user["status"] != "ACTIVE":
        raise HTTPException(403, "Account is not active")

    token = create_access_token(str(user["user_id"]))

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")

    db.execute(
        text("INSERT INTO login_sessions (user_id, ip_address, user_agent, status) "
             "VALUES (:uid, :ip, :ua, 'ACTIVE')"),
        {"uid": str(user["user_id"]), "ip": ip, "ua": ua},
    )
    db.commit()

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": str(user["user_id"]),
            "name": user["name"],
            "email": user["email"],
            "profile_image_url": profile_image_url,
        },
    }


@app.post("/auth/google", tags=["Authentication"])
def google_auth(payload: GoogleAuthRequest, request: Request, db: Session = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(
            payload.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(401, "Invalid Google token")

    email = (idinfo.get("email") or "").lower().strip()
    name = (idinfo.get("name") or email.split("@")[0]).strip()
    picture = idinfo.get("picture")
    email_verified = idinfo.get("email_verified", False)

    if not email:
        raise HTTPException(401, "Google account did not return an email")
    if not email_verified:
        raise HTTPException(401, "Google account email is not verified")

    user = db.execute(
        text("SELECT user_id, name, email, status FROM users WHERE email = :e"),
        {"e": email},
    ).mappings().first()

    if not user:
        user = db.execute(
            text("""
                INSERT INTO users (name, email, password_hash, profile_image_url, status)
                VALUES (:n, :e, NULL, :p, 'ACTIVE')
                RETURNING user_id, name, email, status
            """),
            {"n": name, "e": email, "p": picture},
        ).mappings().first()
        db.commit()
    else:
        db.execute(
            text("UPDATE users SET profile_image_url = :p, updated_at = NOW() "
                 "WHERE user_id = :uid"),
            {"p": picture, "uid": str(user["user_id"])},
        )
        db.commit()

    return _issue_login(db, request, user, profile_image_url=picture)


@app.post("/auth/register", tags=["Authentication"])
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    name = payload.name.strip()

    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Invalid email address")

    existing = db.execute(
        text("SELECT user_id FROM users WHERE email = :e"), {"e": email}
    ).mappings().first()
    if existing:
        raise HTTPException(409, "An account with this email already exists")

    user = db.execute(
        text("""
            INSERT INTO users (name, email, password_hash, status)
            VALUES (:n, :e, :p, 'ACTIVE')
            RETURNING user_id, name, email, status
        """),
        {"n": name, "e": email, "p": hash_password(payload.password)},
    ).mappings().first()
    db.commit()

    return _issue_login(db, request, user)


@app.post("/auth/login", tags=["Authentication"])
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    user = db.execute(
        text("SELECT user_id, name, email, password_hash, status "
             "FROM users WHERE email = :e"),
        {"e": email},
    ).mappings().first()

    if not user or not user["password_hash"] or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    return _issue_login(db, request, user)


@app.post("/auth/forgot-password", tags=["Authentication"])
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Demo-simple password reset: no SMTP is configured in this project, so
    there is no email-verification step — entering the account's email is
    treated as proof enough to set a new password. Good for a dev/test app;
    swap in a real emailed reset-token flow before shipping this for real users.
    """
    email = payload.email.lower().strip()

    user = db.execute(
        text("SELECT user_id, status FROM users WHERE email = :e"),
        {"e": email},
    ).mappings().first()

    if not user:
        raise HTTPException(404, "No account found with that email")
    if user["status"] != "ACTIVE":
        raise HTTPException(403, "Account is not active")

    db.execute(
        text("UPDATE users SET password_hash = :p, updated_at = NOW() WHERE user_id = :uid"),
        {"p": hash_password(payload.new_password), "uid": str(user["user_id"])},
    )
    db.commit()

    return {"success": True, "message": "Password updated. You can now sign in."}


@app.post("/auth/fake", tags=["Authentication"])
def fake_login(request: Request, db: Session = Depends(get_db)):
    if not ALLOW_FAKE_LOGIN:
        raise HTTPException(404, "Not found")

    fake_email = "fake.tester@dataquery.local"

    user = db.execute(
        text("SELECT user_id, name, email, status FROM users WHERE email = :e"),
        {"e": fake_email},
    ).mappings().first()

    if not user:
        user = db.execute(
            text("""
                INSERT INTO users (name, email, password_hash, status)
                VALUES ('Fake Tester', :e, NULL, 'ACTIVE')
                RETURNING user_id, name, email, status
            """),
            {"e": fake_email},
        ).mappings().first()
        db.commit()

    return _issue_login(db, request, user)


@app.post("/auth/guest", tags=["Authentication"])
def guest_login():
    """
    Anonymous trial login: real data, real SQL generation, nothing saved.
    Each call mints a brand-new guest id with a fresh GUEST_MAX_PROMPTS
    allowance — logging out and back in as guest resets the count.
    """
    gid = str(uuid4())
    _guest_prompt_counts[gid] = 0

    token = create_access_token(f"guest:{gid}", extra_claims={"guest": True, "gid": gid})

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": f"guest:{gid}",
            "name": "Guest",
            "email": None,
            "profile_image_url": None,
        },
        "guest_prompts_remaining": GUEST_MAX_PROMPTS,
    }


@app.post("/auth/logout", tags=["Authentication"])
def logout(current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    db.execute(
        text("UPDATE login_sessions SET logout_time = NOW(), status = 'LOGGED_OUT' "
             "WHERE user_id = :uid AND status = 'ACTIVE'"),
        {"uid": str(current_user["user_id"])},
    )
    db.commit()
    return {"success": True, "message": "Logged out successfully"}


@app.get("/auth/me", tags=["Authentication"])
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {
        "user_id": str(current_user["user_id"]),
        "name": current_user["name"],
        "email": current_user["email"],
        "profile_image_url": current_user["profile_image_url"],
        "status": current_user["status"],
        "created_at": current_user["created_at"].isoformat(),
    }


# ============================================================
# SESSION ENDPOINTS
# ============================================================
@app.get("/sessions", tags=["Sessions"])
def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT session_id, login_time, logout_time, last_activity,
                   ip_address, user_agent, status
            FROM login_sessions
            WHERE user_id = :uid
            ORDER BY login_time DESC
        """),
        {"uid": str(current_user["user_id"])},
    ).mappings().all()

    sessions = [
        {
            "session_id": str(r["session_id"]),
            "login_time": r["login_time"].isoformat() if r["login_time"] else None,
            "logout_time": r["logout_time"].isoformat() if r["logout_time"] else None,
            "last_activity": r["last_activity"].isoformat() if r["last_activity"] else None,
            "ip_address": r["ip_address"],
            "user_agent": r["user_agent"],
            "status": r["status"],
        }
        for r in rows
    ]
    return {"sessions": sessions}


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def terminate_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.execute(
        text("SELECT session_id, user_id, status FROM login_sessions WHERE session_id = :sid"),
        {"sid": session_id},
    ).mappings().first()

    if not row:
        raise HTTPException(404, "Session not found")

    if str(row["user_id"]) != str(current_user["user_id"]):
        raise HTTPException(403, "Not allowed")

    db.execute(
        text("UPDATE login_sessions SET logout_time = NOW(), status = 'LOGGED_OUT' "
             "WHERE session_id = :sid"),
        {"sid": session_id},
    )
    db.commit()
    return {"success": True, "message": "Session terminated"}


# ============================================================
# CONVERSATION ENDPOINTS
# ============================================================
@app.post("/conversations", tags=["Conversations"])
def create_conversation(payload: ConversationCreateRequest, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    title = (payload.title or "New Conversation").strip() or "New Conversation"

    row = db.execute(
        text("""
            INSERT INTO conversations (user_id, title)
            VALUES (:uid, :t)
            RETURNING conversation_id, title, created_at, updated_at, status
        """),
        {"uid": str(current_user["user_id"]), "t": title},
    ).mappings().first()
    db.commit()

    return {
        "conversation_id": str(row["conversation_id"]),
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


@app.get("/conversations", tags=["Conversations"])
def list_conversations(current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT conversation_id, title, created_at, updated_at, last_message_at, status
            FROM conversations
            WHERE user_id = :uid AND status != 'DELETED'
            ORDER BY COALESCE(last_message_at, updated_at) DESC
        """),
        {"uid": str(current_user["user_id"])},
    ).mappings().all()

    conversations = [
        {
            "conversation_id": str(r["conversation_id"]),
            "title": r["title"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            "last_message_at": r["last_message_at"].isoformat() if r["last_message_at"] else None,
            "status": r["status"],
        }
        for r in rows
    ]
    return {"conversations": conversations}


def _get_owned_conversation(db: Session, conversation_id: str, user_id: str):
    row = db.execute(
        text("""
            SELECT conversation_id, user_id, title, created_at, updated_at,
                   last_message_at, status
            FROM conversations WHERE conversation_id = :cid
        """),
        {"cid": conversation_id},
    ).mappings().first()

    if not row:
        raise HTTPException(404, "Conversation not found")
    if str(row["user_id"]) != str(user_id):
        raise HTTPException(403, "Not allowed")
    return row


@app.get("/conversations/{conversation_id}", tags=["Conversations"])
def get_conversation(conversation_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    row = _get_owned_conversation(db, conversation_id, str(current_user["user_id"]))
    return {
        "conversation_id": str(row["conversation_id"]),
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None,
    }


@app.patch("/conversations/{conversation_id}", tags=["Conversations"])
def update_conversation(conversation_id: str, payload: ConversationUpdateRequest, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    _get_owned_conversation(db, conversation_id, str(current_user["user_id"]))

    updates = []
    params: Dict[str, Any] = {"cid": conversation_id}

    if payload.title is not None:
        updates.append("title = :t")
        params["t"] = payload.title.strip() or "New Conversation"
    if payload.status is not None:
        if payload.status not in ("ACTIVE", "ARCHIVED", "DELETED"):
            raise HTTPException(400, "Invalid status")
        updates.append("status = :s")
        params["s"] = payload.status

    if not updates:
        raise HTTPException(400, "Nothing to update")

    updates.append("updated_at = NOW()")
    sql = f"UPDATE conversations SET {', '.join(updates)} WHERE conversation_id = :cid"
    db.execute(text(sql), params)
    db.commit()
    return {"success": True, "message": "Conversation updated"}


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
def delete_conversation(conversation_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    _get_owned_conversation(db, conversation_id, str(current_user["user_id"]))
    db.execute(
        text("UPDATE conversations SET status = 'DELETED', updated_at = NOW() WHERE conversation_id = :cid"),
        {"cid": conversation_id},
    )
    db.commit()
    return {"success": True, "message": "Conversation deleted"}


# ============================================================
# MESSAGE / CHAT ENDPOINTS
# ============================================================
@app.get("/conversations/{conversation_id}/messages", tags=["Messages"])
def list_messages(conversation_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    _get_owned_conversation(db, conversation_id, str(current_user["user_id"]))

    rows = db.execute(
        text("""
            SELECT message_id, role, content, message_type, sql_generated,
                   created_at, processing_time_ms
            FROM messages
            WHERE conversation_id = :cid
            ORDER BY created_at ASC
        """),
        {"cid": conversation_id},
    ).mappings().all()

    messages = [
        {
            "message_id": str(r["message_id"]),
            "role": r["role"],
            "content": r["content"],
            "message_type": r["message_type"],
            "sql_generated": r["sql_generated"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "processing_time_ms": r["processing_time_ms"],
        }
        for r in rows
    ]
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/messages/{message_id}", tags=["Messages"])
def get_message(message_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT m.message_id, m.conversation_id, m.user_id, m.role, m.content,
                   m.message_type, m.sql_generated, m.created_at, m.processing_time_ms
            FROM messages m
            WHERE m.message_id = :mid
        """),
        {"mid": message_id},
    ).mappings().first()

    if not row:
        raise HTTPException(404, "Message not found")
    if str(row["user_id"]) != str(current_user["user_id"]):
        raise HTTPException(403, "Not allowed")

    return {
        "message_id": str(row["message_id"]),
        "conversation_id": str(row["conversation_id"]),
        "role": row["role"],
        "content": row["content"],
        "message_type": row["message_type"],
        "sql_generated": row["sql_generated"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "processing_time_ms": row["processing_time_ms"],
    }


@app.post("/conversations/{conversation_id}/chat", tags=["Messages"])
def chat(conversation_id: str, payload: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    start_ts = time.time()
    user_id = str(current_user["user_id"])

    _get_owned_conversation(db, conversation_id, user_id)

    question = payload.message.strip()
    if not question:
        raise HTTPException(400, "Message cannot be empty")

    # P0 #1 — Python-side guard: reply immediately if not a real question
    if not is_likely_data_question(question):
        user_msg_refuse = db.execute(
            text("""
                INSERT INTO messages (conversation_id, user_id, role, content, message_type)
                VALUES (:cid, :uid, 'user', :c, 'text')
                RETURNING message_id
            """),
            {"cid": conversation_id, "uid": user_id, "c": question},
        ).mappings().first()

        assistant_refuse = db.execute(
            text("""
                INSERT INTO messages
                    (conversation_id, user_id, role, content, message_type, processing_time_ms)
                VALUES (:cid, :uid, 'assistant', :c, 'text', :ms)
                RETURNING message_id
            """),
            {
                "cid": conversation_id,
                "uid": user_id,
                "c": REFUSE_MESSAGE,
                "ms": int((time.time() - start_ts) * 1000),
            },
        ).mappings().first()

        db.execute(
            text("UPDATE conversations SET last_message_at = NOW(), updated_at = NOW() "
                 "WHERE conversation_id = :cid"),
            {"cid": conversation_id},
        )
        db.commit()

        return {
            "success": True,
            "conversation_id": conversation_id,
            "user_message": {
                "message_id": str(user_msg_refuse["message_id"]),
                "content": question,
            },
            "assistant_message": {
                "message_id": str(assistant_refuse["message_id"]),
                "content": REFUSE_MESSAGE,
            },
            "query": {
                "query_id": None,
                "sql": None,
                "execution_time_ms": 0,
                "row_count": 0,
            },
            "result": {"columns": [], "rows": []},
            "visualization": {"chart_type": "table", "config": None},
        }

    # 1. Save user message
    user_msg = db.execute(
        text("""
            INSERT INTO messages (conversation_id, user_id, role, content, message_type)
            VALUES (:cid, :uid, 'user', :c, 'text')
            RETURNING message_id, created_at
        """),
        {"cid": conversation_id, "uid": user_id, "c": question},
    ).mappings().first()
    db.commit()

    # 2. Context + schema
    context = load_recent_messages(db, conversation_id)
    schema = get_analytics_schema(db)

    # 3. Generate SQL
    try:
        sql = generate_sql(question, schema, context)
    except Exception as e:
        raise _map_llm_error(e)

    if not sql:
        raise HTTPException(500, "AI returned an empty query. Please try rephrasing.")

    # 3b. REFUSE path (only reachable for first message with no context)
    if sql.strip().upper() == "REFUSE":
        assistant_refuse = db.execute(
            text("""
                INSERT INTO messages
                    (conversation_id, user_id, role, content, message_type, processing_time_ms)
                VALUES (:cid, :uid, 'assistant', :c, 'text', :ms)
                RETURNING message_id
            """),
            {
                "cid": conversation_id,
                "uid": user_id,
                "c": REFUSE_MESSAGE,
                "ms": int((time.time() - start_ts) * 1000),
            },
        ).mappings().first()

        db.execute(
            text("UPDATE conversations SET last_message_at = NOW(), updated_at = NOW() "
                 "WHERE conversation_id = :cid"),
            {"cid": conversation_id},
        )
        db.commit()

        return {
            "success": True,
            "conversation_id": conversation_id,
            "user_message": {
                "message_id": str(user_msg["message_id"]),
                "content": question,
            },
            "assistant_message": {
                "message_id": str(assistant_refuse["message_id"]),
                "content": REFUSE_MESSAGE,
            },
            "query": {
                "query_id": None,
                "sql": None,
                "execution_time_ms": 0,
                "row_count": 0,
            },
            "result": {"columns": [], "rows": []},
            "visualization": {"chart_type": "table", "config": None},
        }

    # 4. Validate SQL (on the raw generated SQL, before LIMIT injection)
    ok, reason = validate_sql(sql)
    if not ok:
        blocked_msg = db.execute(
            text("""
                INSERT INTO messages (conversation_id, user_id, role, content, message_type, sql_generated)
                VALUES (:cid, :uid, 'assistant', :c, 'error', :s)
                RETURNING message_id
            """),
            {
                "cid": conversation_id,
                "uid": user_id,
                "c": f"Query blocked: {reason}",
                "s": sql,
            },
        ).mappings().first()

        db.execute(
            text("""
                INSERT INTO query_executions
                    (message_id, user_id, generated_sql, execution_status, error_message)
                VALUES (:mid, :uid, :s, 'BLOCKED', :e)
            """),
            {
                "mid": str(blocked_msg["message_id"]),
                "uid": user_id,
                "s": sql,
                "e": reason,
            },
        )
        db.commit()
        raise HTTPException(400, "Unsafe SQL query blocked")

    # P1 — enforce row limit before executing
    sql = enforce_row_limit(sql)

    # 5. Execute
    try:
        rows, elapsed_ms = execute_sql(db, sql)
    except Exception as e:
        db.rollback()
        err_msg = db.execute(
            text("""
                INSERT INTO messages (conversation_id, user_id, role, content, message_type, sql_generated)
                VALUES (:cid, :uid, 'assistant', :c, 'error', :s)
                RETURNING message_id
            """),
            {
                "cid": conversation_id,
                "uid": user_id,
                "c": "Unable to execute the query.",
                "s": sql,
            },
        ).mappings().first()

        db.execute(
            text("""
                INSERT INTO query_executions
                    (message_id, user_id, generated_sql, execution_status, error_message)
                VALUES (:mid, :uid, :s, 'FAILED', :e)
            """),
            {
                "mid": str(err_msg["message_id"]),
                "uid": user_id,
                "s": sql,
                "e": str(e),
            },
        )
        db.commit()
        raise HTTPException(400, "Unable to execute the query")

    # 6. Summary + chart
    df = pd.DataFrame(rows)
    summary = generate_summary(question, df, language=payload.language or "en")
    chart_info = pick_chart(df, question)

    # 7. Save assistant message
    total_ms = int((time.time() - start_ts) * 1000)
    assistant_msg = db.execute(
        text("""
            INSERT INTO messages
                (conversation_id, user_id, role, content, message_type, sql_generated, processing_time_ms)
            VALUES (:cid, :uid, 'assistant', :c, 'data_query', :s, :ms)
            RETURNING message_id, created_at
        """),
        {
            "cid": conversation_id,
            "uid": user_id,
            "c": summary,
            "s": sql,
            "ms": total_ms,
        },
    ).mappings().first()

    # 8. Log execution
    qx = db.execute(
        text("""
            INSERT INTO query_executions
                (message_id, user_id, generated_sql, execution_status, execution_time_ms, row_count)
            VALUES (:mid, :uid, :s, 'SUCCESS', :ms, :rc)
            RETURNING query_id, created_at
        """),
        {
            "mid": str(assistant_msg["message_id"]),
            "uid": user_id,
            "s": sql,
            "ms": elapsed_ms,
            "rc": len(rows),
        },
    ).mappings().first()

    # 9. Save result
    db.execute(
        text("""
            INSERT INTO query_results (query_id, result_data, chart_type, chart_config)
            VALUES (:qid, CAST(:rd AS JSONB), :ct, CAST(:cc AS JSONB))
        """),
        {
            "qid": str(qx["query_id"]),
            "rd": json.dumps(rows, default=str),
            "ct": chart_info["chart_type"],
            "cc": json.dumps(chart_info["config"], default=str) if chart_info["config"] else None,
        },
    )

    # 10. Update conversation timestamp
    db.execute(
        text("UPDATE conversations SET last_message_at = NOW(), updated_at = NOW() WHERE conversation_id = :cid"),
        {"cid": conversation_id},
    )
    db.commit()

    columns = list(df.columns) if not df.empty else []

    return {
        "success": True,
        "conversation_id": conversation_id,
        "user_message": {
            "message_id": str(user_msg["message_id"]),
            "content": question,
        },
        "assistant_message": {
            "message_id": str(assistant_msg["message_id"]),
            "content": summary,
        },
        "query": {
            "query_id": str(qx["query_id"]),
            "sql": sql,
            "execution_time_ms": elapsed_ms,
            "row_count": len(rows),
        },
        "result": {
            "columns": columns,
            "rows": rows,
        },
        "visualization": {
            "chart_type": chart_info["chart_type"],
            "config": chart_info["config"],
        },
    }


# ============================================================
# GUEST CHAT — real data, nothing persisted, capped at
# GUEST_MAX_PROMPTS per guest login.
# ============================================================
@app.post("/guest/chat", tags=["Messages"])
def guest_chat(
    payload: GuestChatRequest,
    actor: Dict[str, Any] = Depends(get_guest_actor),
    db: Session = Depends(get_db),
):
    gid = actor["gid"]
    used = _guest_prompt_counts.get(gid, 0)

    if used >= GUEST_MAX_PROMPTS:
        raise HTTPException(
            403,
            "You've used your free guest prompts. Sign in to keep chatting.",
        )

    question = payload.message.strip()
    if not question:
        raise HTTPException(400, "Message cannot be empty")

    def refusal_response():
        return {
            "success": True,
            "conversation_id": None,
            "user_message": {"message_id": str(uuid4()), "content": question},
            "assistant_message": {"message_id": str(uuid4()), "content": REFUSE_MESSAGE},
            "query": {"query_id": None, "sql": None, "execution_time_ms": 0, "row_count": 0},
            "result": {"columns": [], "rows": []},
            "visualization": {"chart_type": "table", "config": None},
            "guest_prompts_remaining": GUEST_MAX_PROMPTS - used,
        }

    # Greetings/noise don't consume a guest prompt.
    if not is_likely_data_question(question):
        return refusal_response()

    schema = get_analytics_schema(db)
    context = [{"role": h.role, "content": h.content[:300]} for h in payload.history[-CONTEXT_MESSAGES:]]

    try:
        sql = generate_sql(question, schema, context)
    except Exception as e:
        raise _map_llm_error(e)

    if not sql:
        raise HTTPException(500, "AI returned an empty query. Please try rephrasing.")

    if sql.strip().upper() == "REFUSE":
        return refusal_response()

    ok, reason = validate_sql(sql)
    if not ok:
        raise HTTPException(400, f"Unsafe SQL query blocked: {reason}")

    sql = enforce_row_limit(sql)

    try:
        rows, elapsed_ms = execute_sql(db, sql)
    except Exception:
        db.rollback()
        raise HTTPException(400, "Unable to execute the query")

    df = pd.DataFrame(rows)
    summary = generate_summary(question, df, language=payload.language or "en")
    chart_info = pick_chart(df, question)
    columns = list(df.columns) if not df.empty else []

    _guest_prompt_counts[gid] = used + 1

    return {
        "success": True,
        "conversation_id": None,
        "user_message": {"message_id": str(uuid4()), "content": question},
        "assistant_message": {"message_id": str(uuid4()), "content": summary},
        "query": {
            "query_id": None,
            "sql": sql,
            "execution_time_ms": elapsed_ms,
            "row_count": len(rows),
        },
        "result": {"columns": columns, "rows": rows},
        "visualization": {
            "chart_type": chart_info["chart_type"],
            "config": chart_info["config"],
        },
        "guest_prompts_remaining": GUEST_MAX_PROMPTS - _guest_prompt_counts[gid],
    }


# ============================================================
# QUERY HISTORY ENDPOINTS
# ============================================================
@app.get("/queries", tags=["Queries"])
def list_queries(current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT query_id, generated_sql, execution_status, execution_time_ms,
                   row_count, error_message, created_at
            FROM query_executions
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 100
        """),
        {"uid": str(current_user["user_id"])},
    ).mappings().all()

    queries = [
        {
            "query_id": str(r["query_id"]),
            "generated_sql": r["generated_sql"],
            "execution_status": r["execution_status"],
            "execution_time_ms": r["execution_time_ms"],
            "row_count": r["row_count"],
            "error_message": r["error_message"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]
    return {"queries": queries}


@app.get("/queries/{query_id}", tags=["Queries"])
def get_query(query_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    qx = db.execute(
        text("""
            SELECT query_id, user_id, generated_sql, execution_status,
                   execution_time_ms, row_count, error_message, created_at
            FROM query_executions WHERE query_id = :qid
        """),
        {"qid": query_id},
    ).mappings().first()

    if not qx:
        raise HTTPException(404, "Query not found")
    if str(qx["user_id"]) != str(current_user["user_id"]):
        raise HTTPException(403, "Not allowed")

    result = db.execute(
        text("""
            SELECT result_id, result_data, chart_type, chart_config, created_at
            FROM query_results WHERE query_id = :qid
            ORDER BY created_at DESC LIMIT 1
        """),
        {"qid": query_id},
    ).mappings().first()

    return {
        "query_id": str(qx["query_id"]),
        "generated_sql": qx["generated_sql"],
        "execution_status": qx["execution_status"],
        "execution_time_ms": qx["execution_time_ms"],
        "row_count": qx["row_count"],
        "error_message": qx["error_message"],
        "created_at": qx["created_at"].isoformat() if qx["created_at"] else None,
        "result": {
            "result_data": result["result_data"] if result else None,
            "chart_type": result["chart_type"] if result else None,
            "chart_config": result["chart_config"] if result else None,
        } if result else None,
    }