# FYP Student Support Chatbot (Trainable + Retrieval + Safety + Memory)

This is a **ready-to-run** Python starter kit you can copy/paste and **alter + train** for your Final Year Project.

It includes:
- ✅ Trainable **intent classifier** (scikit-learn)
- ✅ Trainable **knowledge-base retriever** (TF‑IDF + cosine similarity)
- ✅ **User profile memory** (SQLite) + simple adaptive behaviour
- ✅ **Safety layer** (crisis/self-harm detection + safe response)
- ✅ **FastAPI** backend endpoint (`/chat`)
- ✅ CLI scripts to train + ingest docs

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Train intents + ingest KB
```bash
python src/intents_train.py
python src/kb_ingest.py
```

### 3) Run API
```bash
uvicorn app:app --reload
```

### 4) Test (curl)
```bash
curl -s http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"user_id":"joe","message":"I feel stressed about exams, can you help?"}' | python -m json.tool
```

---

## Your customisation checklist (for marks)
- Add/replace intents in `data/intents.json`
- Add your study guidance documents into `data/kb/*.txt`
- Add evaluation scripts:
  - intent accuracy (train/test split)
  - retrieval precision@k (manual labels or small test set)
- Add a frontend later (optional)

---

## Notes
This is **not medical advice** software. The safety layer helps you stay within ethical boundaries.
