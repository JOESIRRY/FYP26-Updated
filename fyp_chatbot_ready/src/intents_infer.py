from pathlib import Path
import joblib
import numpy as np
from typing import Tuple, Dict

import config

def load_intent_model():
    path = config.INTENT_MODEL_DIR / "intent_pipe.joblib"
    if not path.exists():
        raise FileNotFoundError("Intent model not found. Run: python src/intents_train.py")
    return joblib.load(path)

def predict_intent(model, text: str) -> Tuple[str, float]:
    # Try to use predict_proba if available (LogReg has it)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        tag = model.classes_[idx]
        score = float(probs[idx])
        return tag, score
    tag = model.predict([text])[0]
    return str(tag), 1.0
