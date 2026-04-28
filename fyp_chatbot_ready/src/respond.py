# src/respond.py
"""
Eunoia — Hybrid Response Generator
=====================================
Response priority:
  1. Study planner  → own rule-based NLP engine (always own code)
  2. High-confidence intents → own curated response bank (no API call)
  3. Complex / low confidence → Claude API for natural language
"""
from __future__ import annotations
import re
import random
from typing import Dict, Any, List

from src.retriever import load_kb_index, retrieve_kb
from src.planner import build_study_plan
from src.llm import ollama_generate

SYSTEM_PROMPT = ""
FALLBACK_REPLY = "I'm here 💙 Tell me what's on your mind and I'll do my best to help."

# Confidence threshold for using own responses instead of API
OWN_THRESHOLD = 0.45

# ── Own response bank ──────────────────────────────────────────────────────────

OWN_RESPONSES = {
    "general_chat": [
        "Hey! 👋 I'm Eunoia, your student wellbeing companion. I can help with:\n\n- 📅 **Study planning** — personalised revision timetables\n- 😰 **Exam stress** — anxiety management\n- 😴 **Sleep** — fixing your sleep routine\n- ⏰ **Time management** — beating procrastination\n- 💙 **Mental health** — emotional support\n\nWhat would you like help with today?",
        "Hi there! 😊 I'm Eunoia — here to support you through university life. Whether it's exam stress, sleep problems, motivation, or just needing someone to talk to — I'm here. What's on your mind?",
        "Hello! 👋 Great to meet you. Tell me what you're struggling with and we'll work through it together 💪",
    ],

    "sleep": [
        "😴 Sleep issues are really common among students — you're not alone!\n\nHere's a quick reset plan:\n\n- **Set a consistent wake time** even on weekends\n- **Cut caffeine after 2pm** — it stays in your system for hours\n- **No screens 30 mins before bed** — blue light delays sleep\n- **Wind-down routine** — reading, gentle stretching, journaling\n- **Keep your room cool and dark**\n\nWhat time are you currently going to sleep? I can help you shift it gradually 🌙",
        "Poor sleep affects everything — concentration, mood, memory. Let's fix it! 😴\n\n**The gradual shift method:**\n- Pick your target wake time\n- Move your bedtime earlier by 20-30 mins every 2-3 days\n- This resets your body clock without shocking your system\n\nWhat's your current sleep schedule?",
    ],

    "time_management": [
        "⏰ Procrastination is one of the most common student struggles. Here's what actually works:\n\n- **2-minute rule** — if it takes less than 2 mins, do it now\n- **Pomodoro technique** — 25 mins focused work, 5 min break\n- **Time blocking** — assign specific tasks to specific time slots\n- **Remove distractions** — phone in another room, use website blockers\n- **Start with the hardest task** when your energy is highest\n\nWhat's the main thing you're putting off right now? 🎯",
        "Time management gets easier with a system. Here's a simple one:\n\n1. **Brain dump** — write everything you need to do\n2. **Prioritise** — urgent and important tasks first\n3. **Block time** — assign tasks to specific slots\n4. **Review daily** — check what you did and plan tomorrow\n\nWhat's your biggest time management challenge? ⏰",
    ],

    "motivation": [
        "Lost motivation is completely normal — especially mid-semester. Here's how to get it back:\n\n- **Reconnect with your why** — why did you choose your course?\n- **Set tiny wins** — one small achievable goal per day builds momentum\n- **Change your environment** — study somewhere different\n- **Rest deliberately** — burnout needs real rest, not just stopping work\n- **Talk to someone** — isolation kills motivation\n\nHow long have you been feeling this way? 💙",
        "Burnout is real and your body is telling you something important. Things that genuinely help:\n\n- **Give yourself permission to rest** — guilt makes burnout worse\n- **Do one tiny thing** — momentum builds from small actions\n- **Remember past successes** — you've done hard things before\n- **Speak to your personal tutor** — they can often provide extensions or extra support\n\nWhat used to excite you about your course? 🌱",
    ],

    "exam_anxiety": [
        "Exam anxiety is really common and very treatable 💪 Try this **3-step reset** when panic hits:\n\n1. **Box breathing** — in 4 counts, hold 4, out 4, hold 4\n2. **Grounding** — name 5 things you can see, 4 you can touch, 3 you can hear\n3. **Next action** — pick ONE small thing you can do right now\n\nRemember: anxiety is your body trying to help you perform. Channel it!\n\nWhat specifically are you most worried about?",
        "That exam pressure sounds really tough 😰 Here's what the research says actually helps:\n\n- **Past papers** — the single most effective revision technique\n- **Teach the material** — explain it out loud as if teaching someone else\n- **Spaced repetition** — short sessions over many days beats cramming\n- **Simulate exam conditions** — timed, no notes, proper desk\n\nWould you like me to build you a revision plan? It can make the pressure feel much more manageable 📅",
    ],
}

# ── Plan detection ─────────────────────────────────────────────────────────────

PLAN_EXPLICIT = [
    "study plan", "revision plan", "revision timetable", "study timetable",
    "study schedule", "make me a plan", "build me a plan", "create a plan",
    "make a timetable", "build a timetable", "plan my revision",
    "plan my studying", "help me plan", "i have an exam in",
    "my exam is in", r"exam in \d+ day", r"exam in \d+ week",
    r"\d+ days till", r"\d+ days until", r"\d+ weeks till", r"\d+ weeks until",
]

PLAN_EXCLUSIONS = [
    "revision tips", "revision advice", "how to revise", "tips for",
    "advice on", "how do i", "what should i", "techniques", "strategies",
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

    # ── 1. Study Planner (own code) ──
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

    # ── 2. Own response bank (no API call) ──
    if confidence >= OWN_THRESHOLD and tag in OWN_RESPONSES:
        reply = random.choice(OWN_RESPONSES[tag])
        return {"reply": reply, "intent": {"tag": tag, "confidence": float(confidence)}, "kb_hits": [], "plan": None}

    # ── 3. KB + Claude API (complex messages) ──
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