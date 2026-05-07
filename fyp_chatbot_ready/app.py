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

app = FastAPI(title="Eunoia Student Wellbeing Chatbot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database setup ─────────────────────────────────────────────────────────────

CHAT_DB = config.BASE_DIR / "models" / "chats.sqlite3"

def get_conn():
    CHAT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAT_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, user_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS plans (
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL,
        plan_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, user_id))""")
    conn.commit()
    return conn

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Stable user identifier")
    chat_id: str = Field(..., description="Unique chat session ID")
    message: str = Field(..., min_length=1)
    chat_title: Optional[str] = None

class ChatResponse(BaseModel):
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
    return FileResponse("Frontend/index.html")

@app.get("/planner.html")
def planner_page():
    return FileResponse("Frontend/planner.html")

@app.get("/eunoia-logo.png")
def logo():
    return FileResponse("Frontend/eunoia-logo.png")

# ── Chat route ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conn = get_conn()

    # Upsert chat session
    existing = conn.execute(
        "SELECT chat_id FROM chats WHERE chat_id=? AND user_id=?",
        (req.chat_id, req.user_id)
    ).fetchone()

    title = req.chat_title or req.message[:45]
    if not existing:
        conn.execute(
            "INSERT INTO chats (chat_id,user_id,title,created_at,updated_at) VALUES (?,?,?,?,?)",
            (req.chat_id, req.user_id, title, now_iso(), now_iso())
        )
    else:
        conn.execute(
            "UPDATE chats SET updated_at=? WHERE chat_id=? AND user_id=?",
            (now_iso(), req.chat_id, req.user_id)
        )

    # Save user message
    conn.execute(
        "INSERT INTO messages (chat_id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
        (req.chat_id, req.user_id, "user", req.message, now_iso())
    )
    conn.commit()

    # ── Safety check (always first) ──
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
    current_rows = conn.execute(
        "SELECT content FROM messages WHERE chat_id=? AND user_id=? AND role='user' ORDER BY id ASC",
        (req.chat_id, req.user_id)
    ).fetchall()
    current_msgs = [r[0] for r in current_rows]
    risk = analyse_risk_persistent(CHAT_DB, req.user_id, current_msgs)
    conn.close()

    # ── Intent classification (own ML model) ──
    prof = load_or_create_profile(config.DB_PATH, req.user_id)
    intent_model = load_intent_model()
    tag, conf = predict_intent(intent_model, req.message)

    # ── Generate reply (own planner / KB / LLM) ──
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
        conn2.execute(
            "INSERT OR REPLACE INTO plans (chat_id,user_id,plan_json,created_at) VALUES (?,?,?,?)",
            (req.chat_id, req.user_id, json.dumps(plan_data), now_iso())
        )
    conn2.commit()
    conn2.close()

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
    conn = get_conn()
    rows = conn.execute(
        "SELECT role,content,created_at FROM messages WHERE chat_id=? AND user_id=? ORDER BY id ASC",
        (chat_id, user_id)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]


@app.delete("/history/{user_id}/{chat_id}")
def delete_chat(user_id: str, chat_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM chats WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.execute("DELETE FROM messages WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.execute("DELETE FROM plans WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@app.get("/search/{user_id}")
def search_chats(user_id: str, q: str):
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
    risk = analyse_risk_persistent(CHAT_DB, user_id, [])
    return {
        "risk_score": risk.score,
        "risk_level": risk.level,
        "risk_label": risk.label,
        "risk_color": risk.color,
        "is_crisis": risk.is_crisis
    }