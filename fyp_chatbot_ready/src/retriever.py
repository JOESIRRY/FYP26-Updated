from pathlib import Path
import json
from typing import List, Dict, Any

# Path to the knowledge base index file
DATA_PATH = Path("data/kb.json")

def load_kb_index() -> List[Dict[str, Any]]:
    # Load the knowledge base from disk
    # Returns an empty list if the file doesn't exist yet
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_kb(index: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    # Search the knowledge base for documents relevant to the query
    # Uses simple keyword matching — counts how many query words appear in each document
    q = query.lower().split()
    scored = []

    for item in index:
        text = (item.get("text") or "").lower()
        # Score each document by how many query words it contains
        score = sum(1 for w in q if w in text)
        if score > 0:
            scored.append({
                "doc_id": item.get("doc_id", "kb"),
                "text": item.get("text", ""),
                "score": float(score),
            })

    # Sort by score descending and return the top k results
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]