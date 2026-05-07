"""
Eunoia — Student Wellbeing Chatbot API
========================================
FastAPI backend that integrates:
- Own ML intent classifier (scikit-learn)
- Own ML risk analyser (keyword-weighted NLP)
- Own rule-based study planner
- Own TF-IDF knowledge base retriever
- Claude API for natural language generation
- SQLite for persistent user profiles and chat history
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sqlite3
import json
from pathlib import Path
from datetime import datetime

import config
from src.safety import check_safety, crisis_response
from src.profile import load_or_create_profile
from src.intents_infer import load_intent_model, predict_intent
from src.respond import generate_reply
from src.risk_analyser import analyse_risk_persistent

# Create the FastAPI app instance
app = FastAPI(title="Eunoia Student Wellbeing Chatbot", version="2.0")

# Allow the frontend to communicate with the backend from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database setup ─────────────────────────────────────────────────────────────

# Path to the SQLite database that stores all chats, messages and plans
CHAT_DB = config.BASE_DIR / "models" / "chats.sqlite3"

def get_conn():
    # Connect to SQLite and create all tables if they don't exist yet
    CHAT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAT_DB))

    # Stores each conversation session with its title and timestamps
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, user_id))""")

    # Stores every message sent and received in each conversation
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)""")

    # Stores generated study plans so they persist after page refresh
    conn.execute("""CREATE TABLE IF NOT EXISTS plans (
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL,
        plan_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, user_id))""")

    conn.commit()
    return conn

def now_iso():
    # Returns the current UTC time as an ISO string for use in timestamps
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    # Data the frontend sends when the user submits a message
    user_id: str = Field(..., description="Stable user identifier")
    chat_id: str = Field(..., description="Unique chat session ID")
    message: str = Field(..., min_length=1)
    chat_title: Optional[str] = None

class ChatResponse(BaseModel):
    # Data the backend sends back to the frontend after processing
    reply: str
    intent: Dict[str, Any]
    kb_hits: List[Dict[str, Any]]
    plan: Optional[Dict[str, Any]] = None
    risk_score: int = 5
    risk_level: str = "low"
    risk_label: str = "Good"
    risk_color: str = "#34d399"

# ── Static file routes ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    # Serve the main chat page
    return FileResponse("Frontend/index.html")

@app.get("/planner.html")
def planner_page():
    # Serve the visual study planner page
    return FileResponse("Frontend/planner.html")

@app.get("/eunoia-logo.png")
def logo():
    # Serve the Eunoia logo so it displays correctly in the browser
    return FileResponse("Frontend/eunoia-logo.png")

# ── Chat route ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conn = get_conn()

    # Check if this chat session already exists in the database
    existing = conn.execute(
        "SELECT chat_id FROM chats WHERE chat_id=? AND user_id=?",
        (req.chat_id, req.user_id)
    ).fetchone()

    # Use the first 45 characters of the message as the chat title if none was provided
    title = req.chat_title or req.message[:45]
    if not existing:
        # Create a new chat record for this session
        conn.execute(
            "INSERT INTO chats (chat_id,user_id,title,created_at,updated_at) VALUES (?,?,?,?,?)",
            (req.chat_id, req.user_id, title, now_iso(), now_iso())
        )
    else:
        # Update the last activity timestamp for an existing chat
        conn.execute(
            "UPDATE chats SET updated_at=? WHERE chat_id=? AND user_id=?",
            (now_iso(), req.chat_id, req.user_id)
        )

    # Save the user's message to the database before processing
    conn.execute(
        "INSERT INTO messages (chat_id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
        (req.chat_id, req.user_id, "user", req.message, now_iso())
    )
    conn.commit()

    # ── Safety check (always first) ──
    # Check for crisis language before any ML or API processing
    # If detected, return the crisis response immediately with UK resources
    safety = check_safety(req.message)
    if safety.is_crisis:
        reply = crisis_response("UK")
        conn.execute(
            "INSERT INTO messages (chat_id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
            (req.chat_id, req.user_id, "assistant", reply, now_iso())
        )
        conn.commit()
        conn.close()
        return ChatResponse(
            reply=reply,
            intent={"tag": "crisis", "confidence": 1.0},
            kb_hits=[],
            plan=None,
            risk_score=95,
            risk_level="crisis",
            risk_label="Crisis",
            risk_color="#ef4444"
        )

    # ── Persistent risk analysis across ALL chats ──
    # Fetch all user messages in this chat to pass to the risk analyser
    current_rows = conn.execute(
        "SELECT content FROM messages WHERE chat_id=? AND user_id=? AND role='user' ORDER BY id ASC",
        (req.chat_id, req.user_id)
    ).fetchall()
    current_msgs = [r[0] for r in current_rows]
    # Analyse risk using the current messages plus 14 days of history from all chats
    risk = analyse_risk_persistent(CHAT_DB, req.user_id, current_msgs)
    conn.close()

    # ── Intent classification (own ML model) ──
    # Load the user profile and run the trained intent classifier on the message
    prof = load_or_create_profile(config.DB_PATH, req.user_id)
    intent_model = load_intent_model()
    tag, conf = predict_intent(intent_model, req.message)

    # ── Generate reply (own planner / KB / LLM) ──
    # Route the message to the study planner or Claude API based on the intent
    out = generate_reply(prof, tag, req.message, conf)
    reply = out.get("reply", "")

    # ── Save assistant reply and plan ──
    conn2 = get_conn()
    conn2.execute(
        "INSERT INTO messages (chat_id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
        (req.chat_id, req.user_id, "assistant", reply, now_iso())
    )
    plan_data = out.get("plan")
    if plan_data:
        # Save the generated study plan so it persists even after page refresh
        conn2.execute(
            "INSERT OR REPLACE INTO plans (chat_id,user_id,plan_json,created_at) VALUES (?,?,?,?)",
            (req.chat_id, req.user_id, json.dumps(plan_data), now_iso())
        )
    conn2.commit()
    conn2.close()

    # Return the full response including the reply, intent info, risk score and plan
    return ChatResponse(
        reply=reply,
        intent=out.get("intent", {"tag": tag, "confidence": float(conf)}),
        kb_hits=out.get("kb_hits", []),
        plan=out.get("plan", None),
        risk_score=risk.score,
        risk_level=risk.level,
        risk_label=risk.label,
        risk_color=risk.color,
    )

# ── History routes ─────────────────────────────────────────────────────────────

@app.get("/history/{user_id}")
def get_history(user_id: str):
    # Return all conversations for a user sorted by most recent activity
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.chat_id, c.title, c.created_at, c.updated_at,
           (SELECT content FROM messages WHERE chat_id=c.chat_id AND user_id=c.user_id
            ORDER BY id DESC LIMIT 1) as last_msg
           FROM chats c WHERE c.user_id=? ORDER BY c.updated_at DESC""",
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3], "last_message": r[4]} for r in rows]


@app.get("/history/{user_id}/{chat_id}/messages")
def get_messages(user_id: str, chat_id: str):
    # Return all messages in a specific conversation in chronological order
    conn = get_conn()
    rows = conn.execute(
        "SELECT role,content,created_at FROM messages WHERE chat_id=? AND user_id=? ORDER BY id ASC",
        (chat_id, user_id)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]


@app.delete("/history/{user_id}/{chat_id}")
def delete_chat(user_id: str, chat_id: str):
    # Delete a conversation and all its messages and plans from the database
    conn = get_conn()
    conn.execute("DELETE FROM chats WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.execute("DELETE FROM messages WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.execute("DELETE FROM plans WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@app.get("/search/{user_id}")
def search_chats(user_id: str, q: str):
    # Search through all conversations and messages for a matching keyword
    conn = get_conn()
    rows = conn.execute(
        """SELECT DISTINCT c.chat_id, c.title, c.updated_at
           FROM chats c JOIN messages m ON c.chat_id=m.chat_id AND c.user_id=m.user_id
           WHERE c.user_id=? AND (m.content LIKE ? OR c.title LIKE ?)
           ORDER BY c.updated_at DESC""",
        (user_id, f"%{q}%", f"%{q}%")
    ).fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


@app.get("/plan/{user_id}/{chat_id}")
def get_plan(user_id: str, chat_id: str):
    """Get the saved study plan for a chat."""
    # Retrieve a previously generated study plan from the database
    conn = get_conn()
    row = conn.execute(
        "SELECT plan_json FROM plans WHERE chat_id=? AND user_id=?",
        (chat_id, user_id)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


@app.get("/risk/{user_id}")
def get_global_risk(user_id: str):
    """Get the user's overall wellbeing risk score across all chats."""
    # Used by the frontend on page load to set the initial risk bar state
    risk = analyse_risk_persistent(CHAT_DB, user_id, [])
    return {
        "risk_score": risk.score,
        "risk_level": risk.level,
        "risk_label": risk.label,
        "risk_color": risk.color,
        "is_crisis": risk.is_crisis
    }