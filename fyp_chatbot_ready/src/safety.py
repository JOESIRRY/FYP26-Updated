# src/safety.py
"""
Eunoia Safety Layer — UK-first crisis detection.
Catches crisis messages BEFORE they reach the LLM.
Uses comprehensive keyword patterns for reliable detection.
"""
from dataclasses import dataclass
from typing import Optional
import re

# Comprehensive crisis patterns — covers all common phrasings
_CRISIS_PATTERNS = [
    # Self harm / suicide explicit
    r"\bkill my\s*self\b",
    r"\bkilling my\s*self\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend my life\b",
    r"\btake my own life\b",
    r"\btake my life\b",
    r"\bself harm\b",
    r"\bself-harm\b",
    r"\bselfharm\b",
    r"\bcutting my\s*self\b",
    r"\bi have been cutting\b",
    r"\bhurt my\s*self\b",
    r"\bharming my\s*self\b",
    r"\bwant to die\b",
    r"\bwanting to die\b",
    r"\bwish i was dead\b",
    r"\bwish i were dead\b",
    r"\bno reason to live\b",
    r"\bdon.?t want to be alive\b",
    r"\bdont want to be alive\b",
    r"\bwant to disappear forever\b",
    r"\bend it all\b",
    r"\bnot worth living\b",
    r"\blife is not worth living\b",
    r"\boverdose\b",
    r"\btaken.*pills\b",
    r"\bjump.*bridge\b",
    r"\bthinking about suicide\b",
    r"\bplanning to.*hurt\b",
    r"\bgoing to hurt myself\b",
    # Arabic
    r"\bانتحار\b",
    r"\bاقتل نفسي\b",
    r"\bلا اريد العيش\b",
    r"\bاذى نفسي\b",
]


@dataclass
class SafetyResult:
    is_crisis: bool
    reason: Optional[str] = None


def check_safety(message: str) -> SafetyResult:
    """Check if message contains crisis/self-harm language."""
    text = message.lower().strip()
    for pat in _CRISIS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return SafetyResult(True, "crisis_keywords")
    return SafetyResult(False)


def crisis_response(user_location: str = "UK") -> str:
    """UK-first crisis response with proper resources."""
    return (
        "I'm really sorry you're feeling this way 💙 You don't have to face this alone.\n\n"
        "**Please reach out for immediate support right now:**\n\n"
        "🆘 **Emergency:** Call **999** or go to your nearest **A&E**\n"
        "📞 **Samaritans:** Call **116 123** (free, 24/7, no judgement)\n"
        "💬 **SHOUT Crisis Text Line:** Text **SHOUT** to **85258** (free, 24/7)\n"
        "🏫 **Your university:** Contact your Student Wellbeing / Counselling service\n"
        "🌐 **Mind:** mind.org.uk — mental health support and advice\n\n"
        "**Are you safe right now?** Is there someone nearby you can be with? "
        "I'm here and I care about what happens to you 💙"
    )