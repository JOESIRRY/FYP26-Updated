from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

INTENTS_PATH = DATA_DIR / "intents.json"
KB_DIR = DATA_DIR / "kb"

INTENT_MODEL_DIR = MODELS_DIR / "intent_model"
KB_INDEX_DIR = MODELS_DIR / "kb_index"

DB_PATH = BASE_DIR / "models" / "profiles.sqlite3"

# Retrieval settings
TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.18  # tweak based on your data
