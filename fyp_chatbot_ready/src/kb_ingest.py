from pathlib import Path
from typing import List, Dict
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

import config

def read_docs(kb_dir: Path) -> List[Dict]:
    docs = []
    for p in sorted(kb_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Basic chunking by blank lines for better retrieval granularity
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for j, ch in enumerate(chunks):
            docs.append({"doc_id": f"{p.stem}:{j}", "text": ch})
    return docs

def main() -> None:
    kb_dir = config.KB_DIR
    kb_dir.mkdir(parents=True, exist_ok=True)
    docs = read_docs(kb_dir)
    if not docs:
        raise SystemExit(f"No KB docs found in {kb_dir}. Add .txt files then re-run.")

    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words=None)
    matrix = vectorizer.fit_transform(texts)

    config.KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, config.KB_INDEX_DIR / "kb_vectorizer.joblib")
    joblib.dump(matrix, config.KB_INDEX_DIR / "kb_matrix.joblib")
    joblib.dump(docs, config.KB_INDEX_DIR / "kb_meta.joblib")
    print(f"Ingested {len(docs)} chunks from {kb_dir}")
    print(f"Saved KB index to {config.KB_INDEX_DIR}")

if __name__ == "__main__":
    main()
