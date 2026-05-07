from pathlib import Path
import json
from typing import List, Dict, Any

DATA_PATH = Path("data/kb.json")

def load_kb_index() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_kb(index: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    q = query.lower().split()
    scored = []

    for item in index:
        text = (item.get("text") or "").lower()
        score = sum(1 for w in q if w in text)
        if score > 0:
            scored.append({
                "doc_id": item.get("doc_id", "kb"),
                "text": item.get("text", ""),
                "score": float(score),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]