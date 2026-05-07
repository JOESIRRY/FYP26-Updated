# Eunoia — Student Wellbeing Chatbot

An intelligent web-based chatbot that integrates mental health support with personalised academic guidance for university students. Built as a Final Year Project for BSc Computer Science at the University of East London.

---

## What is Eunoia?

Eunoia is a student wellbeing AI that helps with:

- 😰 **Exam stress and anxiety** — coping strategies and emotional support
- 📅 **Study planning** — generates personalised revision timetables from natural language
- 😴 **Sleep problems** — advice on fixing sleep routines
- ⏰ **Time management** — beating procrastination and staying organised
- 💙 **Mental health** — non-clinical emotional support
- 🆘 **Crisis detection** — immediate UK emergency resources (999, Samaritans, SHOUT)
- 🌍 **Multilingual** — supports Arabic and other languages

---

## System Architecture

The system uses a hybrid ML approach:

- **Intent Classifier** — TF-IDF + Logistic Regression trained on 393 curated examples across 8 intent categories (80.2% cross-validation accuracy)
- **Sentiment Risk Classifier** — TF-IDF + Logistic Regression trained on 16,000 real examples from the HuggingFace dair-ai/emotion dataset (91.8% test accuracy)
- **Study Planner** — rule-based NLP engine that parses natural language and generates day-by-day revision schedules
- **Safety Layer** — keyword-based crisis detection that runs before any ML or API processing
- **Claude API** — Anthropic Claude Haiku for natural language response generation
- **SQLite** — persistent chat history, study plans, and user profiles

---

## Project Structure

```
fyp_chatbot_ready/
├── app.py                  # FastAPI backend — all API routes
├── config.py               # Path configuration
├── requirements.txt        # Python dependencies
├── data/
│   ├── intents.json        # Training data for intent classifier
│   └── kb/                 # Knowledge base documents
├── src/
│   ├── intents_train.py    # Train the intent classifier
│   ├── intents_infer.py    # Run intent classification
│   ├── sentiment_train.py  # Train the sentiment risk classifier
│   ├── evaluate.py         # Evaluate the intent classifier
│   ├── risk_analyser.py    # Two-layer wellbeing risk assessment
│   ├── safety.py           # Crisis detection and UK resources
│   ├── planner.py          # Natural language study planner
│   ├── respond.py          # Response routing and generation
│   ├── retriever.py        # Knowledge base retrieval
│   ├── llm.py              # Claude API integration
│   ├── profile.py          # User profile persistence
│   └── utils.py            # Utility functions
├── models/
│   ├── intent_model/       # Saved intent classifier
│   └── sentiment_model.joblib  # Saved sentiment classifier
└── Frontend/
    ├── index.html          # Main chat interface
    ├── planner.html        # Visual study planner calendar
    └── eunoia-logo.png     # App logo
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JOESIRRY/FYP26-Updated.git
cd FYP26-Updated/fyp_chatbot_ready
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key
```bash
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 5. Train the ML models
```bash
python -m src.intents_train
python -m src.sentiment_train
```

### 6. Start the server
```bash
uvicorn app:app --reload
```

### 7. Open the app
Go to: http://127.0.0.1:8000

---

## Evaluation

Run the intent classifier evaluation to see accuracy, F1 score, confusion matrix and sample predictions:

```bash
python -m src.evaluate
```

**Results:**
- Intent Classifier: **80.2% cross-validation accuracy** across 8 intent categories
- Sentiment Classifier: **91.8% test accuracy** on 16,000 HuggingFace emotion examples
- Crisis detection recall: **100%**

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3, FastAPI, Uvicorn |
| ML Models | scikit-learn (TF-IDF, Logistic Regression) |
| Database | SQLite |
| LLM API | Claude Haiku (Anthropic) |
| Dataset | HuggingFace dair-ai/emotion |
| Frontend | HTML, CSS, JavaScript |

---

## Author

**Youssef Sirry** — Student Number: 2661830
BSc Computer Science, University of East London
Supervisor: Dr Fadi Safieddine
