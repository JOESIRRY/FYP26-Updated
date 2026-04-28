# src/risk_analyser.py
"""
Eunoia Mental Health Risk Analyser
====================================
Two-layer risk assessment system:

Layer 1 — Crisis Detection (rule-based keyword matching)
  Only explicit self-harm / suicidal language triggers Crisis.
  This is intentionally strict — false positives on crisis are dangerous.
  Rule-based is safer and more reliable than ML for this layer.

Layer 2 — Emotional State Tracking (trained ML sentiment model)
  Uses the sentiment classifier trained on 16,000 HuggingFace examples.
  Tracks gradual build-up of distress/anxiety across messages.
  Responds to positive messages by lowering the score.
  A single stressed message does NOT trigger High Risk.
  Sustained distress across multiple messages does.

Risk levels:
  0-25  → LOW    😊 Good     (green)  — calm, neutral, positive
  26-50 → MEDIUM 😟 Moderate (yellow) — some stress, building concern
  51-79 → HIGH   🚨 High Risk (red)   — sustained distress
  80+   → CRISIS 🆘 Crisis   (dark red) — explicit self-harm/suicidal language only
"""

from __future__ import annotations
import re
import sqlite3
from dataclasses import dataclass
from typing import List
from pathlib import Path
from datetime import datetime, timedelta

import config


@dataclass
class RiskResult:
    score: int
    level: str
    label: str
    color: str
    is_crisis: bool = False


# ── Layer 1: Crisis keywords (strict — only explicit self-harm/suicidal) ──────
# These MUST be present to trigger Crisis level
# Exam stress, depression, sadness alone should NEVER trigger this

CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bkilling myself\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend my life\b",
    r"\btake my (own )?life\b",
    r"\bself.?harm\b",
    r"\bcutting myself\b",
    r"\bhave been cutting\b",
    r"\bwant to die\b",
    r"\bwish i was dead\b",
    r"\bno reason to live\b",
    r"\bdon.?t want to be alive\b",
    r"\bwant to disappear forever\b",
    r"\bend it all\b",
    r"\bnot worth living\b",
    r"\bhurt myself\b",
    r"\bharming myself\b",
    r"\btaken.*pills\b",
    r"\boverdose\b",
    # Arabic crisis
    r"\bانتحار\b",
    r"\baقتل نفسي\b",
    r"\bلا اريد العيش\b",
]


def _is_crisis(messages: List[str]) -> bool:
    """Check if any message contains explicit crisis language."""
    for msg in messages:
        text = msg.lower()
        for pattern in CRISIS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    return False


# ── Layer 2: ML sentiment scoring ─────────────────────────────────────────────

_sentiment_model = None

def _load_model():
    global _sentiment_model
    if _sentiment_model is not None:
        return _sentiment_model
    path = config.MODELS_DIR / "sentiment_model.joblib"
    if path.exists():
        import joblib
        _sentiment_model = joblib.load(path)
    return _sentiment_model


def _ml_sentiment_score(messages: List[str]) -> float:
    """
    Score messages using trained sentiment model.
    Returns a value 0-100 representing emotional distress level.
    
    Key behaviours:
    - Single stressed message → low score (10-20)
    - Multiple stressed messages → building score (30-50)
    - Persistent distress across many messages → high score (50-70)
    - Positive messages actively lower the score
    - Short messages (<4 words) are ignored — not enough context
    """
    model = _load_model()
    
    # Filter out short messages — not enough context for reliable sentiment
    valid_messages = [m for m in messages if len(m.split()) >= 4]
    
    if not valid_messages:
        return 0.0

    classes = list(model.classes_) if model else []
    
    # Running score with decay — recent messages matter more
    score = 0.0
    n = len(valid_messages)
    
    for i, msg in enumerate(valid_messages):
        # Recency weight: newest = 1.0, oldest = 0.3
        weight = 0.3 + 0.7 * (i / max(n - 1, 1))
        
        if model:
            try:
                probs = model.predict_proba([msg])[0]
                distress_p = probs[classes.index("distress")] if "distress" in classes else 0
                anxious_p  = probs[classes.index("anxious")]  if "anxious"  in classes else 0
                neutral_p  = probs[classes.index("neutral")]  if "neutral"  in classes else 0
                
                # Per-message contribution
                # distress is more serious than anxious
                # neutral/positive actively reduces score
                msg_contribution = (distress_p * 35) + (anxious_p * 20) - (neutral_p * 15)
                score += msg_contribution * weight
                
            except Exception:
                pass
        
    # Sustained distress amplifier — if many messages are distressed
    # this builds naturally through the weighted sum above
    # No artificial multiplier needed
    
    # Cap at 70 for ML layer — Crisis (80+) only from keyword layer
    return min(70.0, max(0.0, score))


def analyse_risk(messages: List[str]) -> RiskResult:
    """
    Analyse a list of user messages using the two-layer system.
    """
    if not messages:
        return _make_result(5)

    # Layer 1: Crisis check (fast, rule-based, strict)
    if _is_crisis(messages):
        return RiskResult(
            score=90,
            level="crisis",
            label="Crisis",
            color="#ef4444",
            is_crisis=True
        )

    # Layer 2: ML sentiment scoring
    model = _load_model()
    if model:
        score = _ml_sentiment_score(messages)
    else:
        # Fallback: simple keyword counting if model not available
        score = _keyword_fallback(messages)

    return _make_result(int(score))


def _keyword_fallback(messages: List[str]) -> float:
    """Simple keyword fallback if ML model not trained yet."""
    MILD = [r"\bstressed\b", r"\bworried\b", r"\bnervous\b", r"\banxious\b", r"\bstruggling\b"]
    HIGH = [r"\bdepressed\b", r"\boverwhelmed\b", r"\bcant cope\b", r"\bpanicking\b"]
    POSITIVE = [r"\bhappy\b", r"\bgreat\b", r"\bfeeling better\b", r"\bconfident\b"]

    score = 0.0
    valid = [m for m in messages if len(m.split()) >= 4]
    n = len(valid)
    
    for i, msg in enumerate(valid):
        w = 0.3 + 0.7 * (i / max(n - 1, 1))
        t = msg.lower()
        msg_score = 0
        for p in HIGH:
            if re.search(p, t): msg_score += 15
        for p in MILD:
            if re.search(p, t): msg_score += 8
        for p in POSITIVE:
            if re.search(p, t): msg_score -= 10
        score += msg_score * w

    return min(70.0, max(0.0, score))


def analyse_risk_persistent(db_path: Path, user_id: str, current_messages: List[str]) -> RiskResult:
    """
    Analyse risk across chat history.
    IMPORTANT: Crisis check always runs on current messages first
    so a single crisis message triggers Crisis immediately.
    """
    # Always check current messages for crisis FIRST
    # This ensures a single crisis message triggers immediately
    if current_messages and _is_crisis(current_messages):
        return RiskResult(score=90, level="crisis", label="Crisis", color="#ef4444", is_crisis=True)

    # For sentiment scoring, use historical context too
    all_messages = list(current_messages)
    try:
        conn = sqlite3.connect(str(db_path))
        cutoff = (datetime.utcnow() - timedelta(days=14)).isoformat()
        rows = conn.execute(
            """SELECT content FROM messages
               WHERE user_id=? AND role='user' AND created_at > ?
               ORDER BY created_at ASC""",
            (user_id, cutoff)
        ).fetchall()
        conn.close()
        historical = [r[0] for r in rows]
        all_messages = historical + current_messages
        # Also check historical for crisis
        if _is_crisis(historical):
            return RiskResult(score=90, level="crisis", label="Crisis", color="#ef4444", is_crisis=True)
    except Exception:
        pass

    return analyse_risk(all_messages)


def _make_result(score: int) -> RiskResult:
    score = max(0, min(100, score))
    if score >= 80:
        return RiskResult(score=score, level="crisis", label="Crisis",    color="#ef4444", is_crisis=True)
    elif score >= 51:
        return RiskResult(score=score, level="high",   label="High Risk", color="#f87171")
    elif score >= 26:
        return RiskResult(score=score, level="medium", label="Moderate",  color="#fbbf24")
    else:
        return RiskResult(score=score, level="low",    label="Good",      color="#34d399")