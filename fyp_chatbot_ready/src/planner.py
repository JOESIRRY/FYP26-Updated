"""
Eunoia Study Planner
======================
Parses natural language study plan requests and generates
a detailed, time-blocked revision schedule.

Handles inputs like:
  - "I have a week till my exam in Maths and Physics"
  - "exam in 3 days, modules: AI, Networks, 2 hours a day"
  - "help me plan for my exam on 2026-05-10, modules: FYP, 3 hrs/day"
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re


# ── Session type definitions ────────────────────────────────────────────────────

SESSION_TYPES = {
    "Learn":          {"emoji": "📖", "color": "#6366f1", "tip": "Read through your notes and key concepts. Focus on understanding, not memorising."},
    "Active Recall":  {"emoji": "🧠", "color": "#8b5cf6", "tip": "Close your notes and write down everything you remember. Then check what you missed."},
    "Past Papers":    {"emoji": "📝", "color": "#a78bfa", "tip": "Do past exam questions under timed conditions. This is the most effective revision technique."},
    "Review Gaps":    {"emoji": "🔍", "color": "#ec4899", "tip": "Go over the answers you got wrong. Focus only on your weak areas."},
    "Mind Map":       {"emoji": "🗺️", "color": "#f59e0b", "tip": "Create a visual summary connecting key ideas. Great for seeing the big picture."},
    "Flashcards":     {"emoji": "🃏", "color": "#10b981", "tip": "Make and review flashcards for key facts, definitions and formulas."},
    "Practice Test":  {"emoji": "✍️", "color": "#ef4444", "tip": "Simulate exam conditions: no notes, timed, full questions."},
    "Rest & Review":  {"emoji": "☕", "color": "#64748b", "tip": "Light revision only. Read notes, no intense work. Rest is essential for memory consolidation."},
}

TIME_BLOCKS = [
    ("09:00", "Morning"),
    ("11:00", "Late Morning"),
    ("14:00", "Afternoon"),
    ("16:00", "Late Afternoon"),
    ("18:00", "Evening"),
    ("20:00", "Night"),
]


@dataclass
class PlanItem:
    day: str           # YYYY-MM-DD
    day_label: str     # "Monday 21 Apr"
    start_time: str    # HH:MM
    end_time: str      # HH:MM
    duration_min: int
    session_type: str
    emoji: str
    color: str
    module: str
    title: str
    tip: str
    is_rest_day: bool = False


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_days_until(text: str) -> Optional[int]:
    """Parse many natural language time expressions."""
    t = text.lower()

    # "is in X days/weeks" or "in X days/weeks"
    m = re.search(r"\b(?:is\s+)?in\s+(\d+)\s*(days?|weeks?)\b", t)
    if m:
        n = int(m.group(1))
        return n * 7 if "week" in m.group(2) else n

    # "X days/weeks till/until/before/away/left"
    m = re.search(r"(\d+)\s*(days?|weeks?)\s*(till|until|before|left|away|to go|time)", t)
    if m:
        n = int(m.group(1))
        return n * 7 if "week" in m.group(2) else n

    # word numbers: "a week", "two weeks", "three days" etc.
    word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"a":1,"an":1}
    m = re.search(r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(weeks?|days?)\b", t)
    if m:
        n = word_nums.get(m.group(1), 1)
        return n * 7 if "week" in m.group(2) else n

    if "next week" in t:
        return 7
    if "a few days" in t:
        return 3
    if "tomorrow" in t:
        return 1

    return None


def _parse_deadline_date(text: str) -> Optional[date]:
    """Parse explicit date like 2026-05-10 or 10/05/2026."""
    # ISO format
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # UK format dd/mm/yyyy
    m = re.search(r"(\d{1,2})/(\d{1,2})/(20\d{2})", text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    return None


def _parse_hours_per_day(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", text.lower())
    return float(m.group(1)) if m else None


def _parse_start_time(text: str) -> Optional[str]:
    """Parse a specific start time from natural language.
    e.g. 'study at 2pm' -> '14:00', 'start at 9am' -> '09:00'
    """
    t = text.lower()
    # Look for "study/start at X" or "from X"
    m = re.search(
        r"(?:study|start|begin|from|at)\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        t, re.I
    )
    if not m:
        # Just look for any time mention
        m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t, re.I)

    if m:
        hour = int(m.group(1))
        mins = int(m.group(2)) if m.group(2) else 0
        period = m.group(3).lower()
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{mins:02d}"
    return None


def _parse_modules(text: str) -> List[str]:
    """Extract module names from text - handles many natural language patterns."""

    STOP = {"my", "the", "a", "an", "i", "have", "got", "this", "that",
            "upcoming", "big", "important", "can", "you", "make",
            "me", "planner", "please", "help", "build", "create", "for"}

    def strip_stops(words):
        while words and words[0].lower() in STOP:
            words = words[1:]
        return words

    # 1. Explicit "modules: X, Y, Z"
    m = re.search(r"modules?\s*[:\-]\s*([^\n.]+)", text, re.IGNORECASE)
    if m:
        raw = re.split(r"\b(in\s+\d+|hours?|hrs?|days?|weeks?|i\s+have|\d+\s+hours?|but|each)\b",
                       m.group(1), flags=re.I)[0]
        parts = [p.strip(" .,") for p in re.split(r",|;|\||\s+and\s+", raw, flags=re.I) if p.strip()]
        parts = [p for p in parts if p.lower() not in STOP and 1 < len(p) <= 35]
        if parts:
            return parts[:6]

    # 2. "[optional words] SUBJECT exam" - grab last 1-4 words before "exam"
    m = re.search(r"\b((?:\w+\s+){0,4}\w+)\s+exam\b", text, re.IGNORECASE)
    if m:
        words = strip_stops(m.group(1).strip().split())
        subject = " ".join(words).strip()
        if subject and 1 < len(subject) <= 35:
            return [subject]

    # 3. "exam in/for/on Maths and Physics"
    m = re.search(r"\b(?:exam|exams|test|module|subject|revision)\s+(?:in|for|on)\s+([^,.\n]+)",
                  text, re.IGNORECASE)
    if m:
        raw = re.split(r"\b(in\s+\d+|hours?|hrs?|days?|weeks?|\d+\s+hours?|but|each)\b",
                       m.group(1), flags=re.I)[0]
        parts = [p.strip() for p in re.split(r",|;|\s+and\s+", raw, flags=re.I) if p.strip()]
        parts = [p for p in parts if p.lower() not in STOP and 1 < len(p) <= 35]
        if parts:
            return parts[:6]

    return []

def _get_session_sequence(days_available: int, n_modules: int) -> List[str]:
    """
    Build a smart session sequence based on how many days are available.
    More days = more variety. Close to exam = more practice tests.
    """
    if days_available <= 2:
        # Last minute: rapid fire
        sequence = ["Active Recall", "Past Papers", "Review Gaps", "Practice Test"]
    elif days_available <= 4:
        # Short prep
        sequence = ["Learn", "Active Recall", "Past Papers", "Review Gaps", "Practice Test"]
    elif days_available <= 7:
        # One week
        sequence = ["Learn", "Learn", "Active Recall", "Past Papers", "Review Gaps", "Mind Map", "Practice Test"]
    else:
        # Multiple weeks
        sequence = ["Learn", "Learn", "Mind Map", "Active Recall", "Flashcards", "Past Papers", "Review Gaps", "Practice Test"]

    return sequence


def _format_day_label(d: date) -> str:
    return d.strftime("%A %-d %b")


def _add_minutes(time_str: str, minutes: int) -> str:
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def build_study_plan(message: str, default_start_time: str = "09:00") -> Tuple[Optional[Dict], Optional[str]]:
    """
    Build a detailed study plan from natural language input.
    Returns (plan_dict, error_message).
    """

    # Parse inputs
    modules = _parse_modules(message)
    hours = _parse_hours_per_day(message)
    days_until = _parse_days_until(message)
    deadline_date = _parse_deadline_date(message)

    # Parse specific start time if provided
    parsed_time = _parse_start_time(message)
    if parsed_time:
        default_start_time = parsed_time

    # Determine deadline
    if deadline_date:
        days_until = (deadline_date - date.today()).days
    elif days_until:
        deadline_date = date.today() + timedelta(days=days_until)
    else:
        deadline_date = None
        days_until = None

    # Collect what's missing
    missing = []
    if not modules:
        missing.append("your module(s) — e.g. *Modules: Maths, Physics*")
    if not hours:
        hours = 2.0  # sensible default
    if deadline_date is None:
        missing.append("when your exam is — e.g. *exam in 1 week* or *exam on 2026-05-10*")

    if missing:
        return None, (
            "I'd love to build your study plan! I just need a couple of things:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nFor example: *'I have an exam in Maths and Physics in 5 days, 3 hours a day'*"
        )

    # Cap duration
    duration_min = max(45, min(180, int(hours * 60)))
    days_available = max(1, days_until)

    # Build schedule
    session_seq = _get_session_sequence(days_available, len(modules))
    items: List[PlanItem] = []
    session_idx = 0
    mod_idx = 0

    today = date.today()

    for day_offset in range(days_available):
        d = today + timedelta(days=day_offset + 1)
        if d > deadline_date:
            break

        days_left = (deadline_date - d).days
        is_last_day = days_left == 0
        is_pre_exam = days_left == 1

        if is_last_day:
            # Exam day — light review only
            items.append(PlanItem(
                day=d.isoformat(),
                day_label=_format_day_label(d) + " 🎓 EXAM DAY",
                start_time="08:00",
                end_time="09:00",
                duration_min=60,
                session_type="Rest & Review",
                emoji="🎓",
                color="#64748b",
                module="All Modules",
                title="Light review + confidence building",
                tip="Go over key summaries only. Eat well, sleep early the night before. You've got this! 💪",
                is_rest_day=True
            ))
            continue

        if is_pre_exam:
            # Day before exam — rest and light review
            items.append(PlanItem(
                day=d.isoformat(),
                day_label=_format_day_label(d) + " — Eve of Exam",
                start_time="10:00",
                end_time="11:30",
                duration_min=90,
                session_type="Rest & Review",
                emoji="☕",
                color="#64748b",
                module="All Modules",
                title="Light revision + rest",
                tip="Review key flashcards and mind maps. No new material. Rest well tonight — sleep is crucial for memory. 🌙",
                is_rest_day=True
            ))
            continue

        # Regular study day — can have multiple sessions if hours allow
        sessions_today = max(1, round(hours / 1.5))
        start_time = default_start_time if hours <= 3 else "09:00"

        for s in range(sessions_today):
            if s > 0:
                # Add a break between sessions
                start_time = _add_minutes(start_time, duration_min // sessions_today + 30)

            stype = session_seq[session_idx % len(session_seq)]
            sinfo = SESSION_TYPES[stype]
            mod = modules[mod_idx % len(modules)]

            end_time = _add_minutes(start_time, duration_min // sessions_today)

            items.append(PlanItem(
                day=d.isoformat(),
                day_label=_format_day_label(d),
                start_time=start_time,
                end_time=end_time,
                duration_min=duration_min // sessions_today,
                session_type=stype,
                emoji=sinfo["emoji"],
                color=sinfo["color"],
                module=mod,
                title=f"{sinfo['emoji']} {mod}: {stype}",
                tip=sinfo["tip"],
            ))

            session_idx += 1
            mod_idx += 1

    plan = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {
            "modules": modules,
            "hours_per_day": hours,
            "days_available": days_available,
            "deadline_or_exam": deadline_date.isoformat(),
            "default_start_time": default_start_time,
        },
        "summary": {
            "total_sessions": len(items),
            "total_hours": round(sum(i.duration_min for i in items) / 60, 1),
            "days_until_exam": days_until,
        },
        "items": [
            {
                "day": it.day,
                "day_label": it.day_label,
                "start_time": it.start_time,
                "end_time": it.end_time,
                "duration_min": it.duration_min,
                "session_type": it.session_type,
                "emoji": it.emoji,
                "color": it.color,
                "module": it.module,
                "title": it.title,
                "tip": it.tip,
                "is_rest_day": it.is_rest_day,
            }
            for it in items
        ],
    }

    return plan, None