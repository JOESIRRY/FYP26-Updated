# src/intents_train.py
"""
Eunoia — Intent Classifier Training
======================================
Trains a TF-IDF + Logistic Regression pipeline on intents.json.
Uses optimised hyperparameters for better accuracy.
"""
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import numpy as np

import config
from src.utils import load_json


def balance_dataset(X, y):
    """Oversample minority classes to balance the dataset."""
    from collections import Counter
    counts = Counter(y)
    max_count = max(counts.values())

    X_bal, y_bal = list(X), list(y)
    for label, count in counts.items():
        if count < max_count:
            indices = [i for i, yi in enumerate(y) if yi == label]
            X_class = [X[i] for i in indices]
            y_class = [y[i] for i in indices]
            X_up, y_up = resample(X_class, y_class, n_samples=max_count - count, random_state=42)
            X_bal.extend(X_up)
            y_bal.extend(y_up)

    return X_bal, y_bal


def main() -> None:
    data = load_json(config.INTENTS_PATH)
    intents = data["intents"]

    X, y = [], []
    for it in intents:
        tag = it["tag"]
        for ex in it["examples"]:
            X.append(ex)
            y.append(tag)

    print(f"Total examples: {len(X)} across {len(set(y))} intents")

    # Balance the dataset
    X_bal, y_bal = balance_dataset(X, y)
    print(f"After balancing: {len(X_bal)} examples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # Optimised pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial"
        ))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {round(acc, 4)} ({acc*100:.1f}%)")
    print(classification_report(y_test, preds, zero_division=0))

    config.INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.INTENT_MODEL_DIR / "intent_pipe.joblib")
    print(f"Saved model to {config.INTENT_MODEL_DIR}")


if __name__ == "__main__":
    main()