import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

@dataclass
class UserProfile:
    user_id: str
    created_at: str
    last_seen: str
    preferences: Dict[str, str]
    topic_weights: Dict[str, float]  # simple adaptivity

def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            preferences_json TEXT NOT NULL,
            topic_weights_json TEXT NOT NULL
        )
        """
    )
    conn.commit()

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def load_or_create_profile(db_path: Path, user_id: str) -> UserProfile:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn)

    row = conn.execute(
        "SELECT user_id, created_at, last_seen, preferences_json, topic_weights_json FROM profiles WHERE user_id=?",
        (user_id,),
    ).fetchone()

    import json
    if row:
        prof = UserProfile(
            user_id=row[0],
            created_at=row[1],
            last_seen=row[2],
            preferences=json.loads(row[3]),
            topic_weights=json.loads(row[4]),
        )
        # update last_seen
        prof.last_seen = _now()
        conn.execute(
            "UPDATE profiles SET last_seen=? WHERE user_id=?",
            (prof.last_seen, user_id),
        )
        conn.commit()
        conn.close()
        return prof

    prof = UserProfile(
        user_id=user_id,
        created_at=_now(),
        last_seen=_now(),
        preferences={"tone": "friendly_uk"},
        topic_weights={"study_plan": 1.0, "exam_anxiety": 1.0, "sleep": 1.0, "general_chat": 1.0},
    )
    conn.execute(
        "INSERT INTO profiles (user_id, created_at, last_seen, preferences_json, topic_weights_json) VALUES (?,?,?,?,?)",
        (prof.user_id, prof.created_at, prof.last_seen, json.dumps(prof.preferences), json.dumps(prof.topic_weights)),
    )
    conn.commit()
    conn.close()
    return prof

def bump_topic_weight(db_path: Path, user_id: str, tag: str, bump: float = 0.05, cap: float = 2.0) -> None:
    import json
    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT topic_weights_json FROM profiles WHERE user_id=?",
        (user_id,),
    ).fetchone()
    if not row:
        conn.close()
        return
    weights = json.loads(row[0])
    weights[tag] = min(cap, float(weights.get(tag, 1.0)) + bump)
    conn.execute(
        "UPDATE profiles SET topic_weights_json=? WHERE user_id=?",
        (json.dumps(weights), user_id),
    )
    conn.commit()
    conn.close()
