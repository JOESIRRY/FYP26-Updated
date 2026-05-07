from pathlib import Path
import joblib
import numpy as np
from typing import Tuple, Dict

import config

def load_intent_model():
    # Load the trained intent classifier from disk
    # Raises an error if the model hasn't been trained yet
    path = config.INTENT_MODEL_DIR / "intent_pipe.joblib"
    if not path.exists():
        raise FileNotFoundError("Intent model not found. Run: python src/intents_train.py")
    return joblib.load(path)

def predict_intent(model, text: str) -> Tuple[str, float]:
    # Run the intent classifier on a single message
    # Returns the predicted intent tag and the confidence score

    # Try to use predict_proba if available (LogReg has it)
    # This gives us a confidence score alongside the prediction
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        idx = int(np.argmax(probs))        # Index of the highest probability
        tag = model.classes_[idx]          # Intent label for that index
        score = float(probs[idx])          # Confidence score between 0 and 1
        return tag, score

    # Fallback for models without probability support
    tag = model.predict([text])[0]
    return str(tag), 1.0