"""
Eunoia — Response Generator
==============================
Simple and clean:
  1. Study planner requests → own rule-based NLP engine (no API)
  2. Everything else → Claude API (natural, accurate, never repetitive)
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

from src.retriever import load_kb_index, retrieve_kb
from src.planner import build_study_plan
from src.llm import ollama_generate

SYSTEM_PROMPT = ""
FALLBACK_REPLY = "I'm here 💙 Tell me what's on your mind and I'll do my best to help."

PLAN_EXPLICIT = [
    "study plan", "revision plan", "revision timetable", "study timetable",
    "study schedule", "make me a plan", "build me a plan", "create a plan",
    "make a timetable", "build a timetable", "plan my revision",
    "plan my studying", "help me plan", "i have an exam in",
    "my exam is in", "exam in", "days till", "days until",
    "weeks till", "weeks until", "days before my exam",
]

PLAN_EXCLUSIONS = [
    "revision tips", "revision advice", "how to revise", "tips for",
    "advice on", "how do i", "what should i", "techniques",
    "how to study", "study tips", "study advice",
]


def _wants_plan(tag: str, message: str) -> bool:
    msg_l = message.lower().strip()
    for excl in PLAN_EXCLUSIONS:
        if excl in msg_l:
            return False
    if tag == "study_plan":
        has_time = bool(re.search(r"\d+\s*(days?|weeks?)|next week|tomorrow|a week", msg_l))
        has_explicit = any(kw in msg_l for kw in ["make", "build", "create", "plan", "timetable", "schedule"])
        if has_time or has_explicit:
            return True
    for kw in PLAN_EXPLICIT:
        if re.search(kw, msg_l):
            return True
    return False


def generate_reply(profile: Dict[str, Any], tag: str, message: str, confidence: float) -> Dict[str, Any]:

    # ── 1. Study Planner (own code, no API) ──
    if _wants_plan(tag, message):
        plan, err = build_study_plan(message, default_start_time="09:00")
        if err:
            reply = ollama_generate(SYSTEM_PROMPT, message)
            if not reply:
                reply = "I'd love to build you a study plan! I just need:\n\n- 📚 **Your module(s)** — e.g. Maths, Biology\n- 📅 **When your exam is** — e.g. *in 5 days*, *next week*\n- ⏰ **Hours per day** you can study\n\nFor example: *'Maths exam in 5 days, 2 hours a day'*"
            return {"reply": reply, "intent": {"tag": "study_plan", "confidence": float(confidence)}, "kb_hits": [], "plan": None}

        s = plan.get("summary", {})
        inp = plan.get("inputs", {})
        modules = inp.get("modules", [])
        mod_text = ", ".join(modules) if modules else "your modules"
        reply = (
            f"✅ I've built your study plan!\n\n"
            f"📚 **Modules:** {mod_text}\n"
            f"📅 **{s.get('days_until_exam', '?')} days** until your exam\n"
            f"🎯 **{s.get('total_sessions', len(plan.get('items', [])))} sessions** · {s.get('total_hours', '?')} total study hours\n\n"
            f"Click **View Full Planner** to see your day-by-day schedule with session tips 💪"
        )
        return {"reply": reply, "intent": {"tag": "study_plan", "confidence": float(confidence)}, "kb_hits": [], "plan": plan}

    # ── 2. Claude API for everything else ──
    kb_hits_out: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    try:
        index = load_kb_index()
        hits = retrieve_kb(index, message, top_k=3)
        for h in hits:
            kb_hits_out.append({"doc_id": h.get("doc_id"), "score": float(h.get("score", 0.0))})
            snippet = (h.get("text") or "").strip()
            if snippet:
                context_lines.append(f"- {snippet}")
    except Exception:
        pass

    reply = ollama_generate(SYSTEM_PROMPT, message, "\n".join(context_lines))
    if not reply:
        reply = FALLBACK_REPLY

    return {"reply": reply, "intent": {"tag": tag, "confidence": float(confidence)}, "kb_hits": kb_hits_out, "plan": None}