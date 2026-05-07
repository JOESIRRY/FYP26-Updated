"""
LLM integration for Eunoia using Claude API.
Note for FYP: The LLM handles natural language generation only.
The project's own ML components handle intent classification,
risk analysis, study planning, and KB retrieval.
"""
import os
import urllib.request
import json

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are Eunoia, a warm and supportive student wellbeing chatbot for UK university students.

LOCATION: You are based in the UK. Always prioritise UK resources, services, and terminology:
- Use "university" not "college", "modules" not "classes", "marks" not "grades" where appropriate
- For crisis/mental health resources, ALWAYS mention UK services first: Samaritans (116 123), SHOUT (text 85258), Student Minds, university counselling
- Never lead with US resources (e.g. 988 Hotline)

LANGUAGE: Always respond in the same language the student writes in. Arabic, French, Spanish, any language — respond in kind. Never refuse a message just because of language.

SCOPE: You help UK university students with:
- Study planning, revision strategies, exam preparation
- Exam stress, anxiety, and academic pressure
- Mental health and emotional wellbeing (non-clinical support)
- Sleep habits, burnout, and lifestyle
- Time management, procrastination, motivation
- General student life challenges

If asked about something completely unrelated to student life (e.g. sports scores, cooking recipes, celebrity news), politely explain you're focused on student wellbeing and redirect.
Language is NEVER a reason to redirect — a student writing in Arabic about exam stress should get a full Arabic response.

TONE:
- Warm, non-judgmental, encouraging
- Use emojis naturally to make responses feel human
- Keep responses focused and actionable — not too long
- Bold key points using **bold**
- Use bullet points for lists starting with -
- Never give medical diagnoses — always encourage professional help for serious concerns
- If a student seems in serious distress, always signpost UK mental health support

FORMATTING:
- Use **bold** for emphasis
- Use - for bullet lists
- Keep paragraphs short (2-3 sentences max)
- Separate sections with a blank line
"""


def ollama_generate(system: str, user: str, context: str = "") -> str:
    """
    Generate a response using the Claude API.
    The 'system' parameter is ignored (we use our own SYSTEM_PROMPT).
    """
    if not ANTHROPIC_API_KEY:
        return "API key not set. Please set ANTHROPIC_API_KEY environment variable."

    full_user = user
    if context:
        full_user = f"Relevant knowledge base context:\n{context}\n\nStudent message:\n{user}"

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": full_user}
        ]
    }

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()
    except Exception as e:
        return f"Sorry, I couldn't get a response right now. Please try again in a moment. ({str(e)})"