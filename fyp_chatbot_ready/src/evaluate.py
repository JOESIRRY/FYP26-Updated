"""
Eunoia — Model Evaluation Script
===================================
Evaluates the intent classifier performance.

Run with:
    python -m src.evaluate
"""
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import resample
import joblib

import config
from src.utils import load_json


def load_data():
    data = load_json(config.INTENTS_PATH)
    X, y = [], []
    for intent in data["intents"]:
        for example in intent["examples"]:
            X.append(example)
            y.append(intent["tag"])
    return X, y


def balance_dataset(X, y):
    from collections import Counter
    counts = Counter(y)
    max_count = max(counts.values())
    X_bal, y_bal = list(X), list(y)
    for label, count in counts.items():
        if count < max_count:
            indices = [i for i, yi in enumerate(y) if yi == label]
            Xc = [X[i] for i in indices]
            yc = [y[i] for i in indices]
            Xu, yu = resample(Xc, yc, n_samples=max_count - count, random_state=42)
            X_bal.extend(Xu)
            y_bal.extend(yu)
    return X_bal, y_bal


def evaluate_intent_classifier():
    print("=" * 60)
    print("  Eunoia — Intent Classifier Evaluation")
    print("=" * 60)

    X, y = load_data()
    print(f"\n📊 Dataset: {len(X)} examples across {len(set(y))} intents")

    from collections import Counter
    print("\nIntent distribution:")
    for tag, count in sorted(Counter(y).items()):
        print(f"  {tag:<20} {count} examples")

    X_bal, y_bal = balance_dataset(X, y)
    print(f"\n⚖️  Balanced dataset: {len(X_bal)} examples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True, max_features=5000)),
        ("clf", LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced", solver="lbfgs"))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"\n{'─'*60}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Weighted F1    : {f1:.4f}")
    print(f"{'─'*60}")

    print("\n📋 Per-Intent Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    cv_scores = cross_val_score(pipe, X_bal, y_bal, cv=5, scoring="accuracy")
    print(f"🔁 5-Fold Cross-Validation:")
    print(f"  Scores : {[round(s, 3) for s in cv_scores]}")
    print(f"  Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\n🔲 Confusion Matrix:")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, preds, labels=labels)
    print(f"  Labels: {labels}")
    for i, row in enumerate(cm):
        print(f"  {labels[i]:<20} {list(row)}")

    print(f"\n🧪 Sample Predictions:")
    test_msgs = [
        "I'm really stressed about my exams",
        "can you make me a study timetable",
        "I can't sleep at night",
        "I feel really depressed lately",
        "hello there",
        "I keep procrastinating and wasting time",
        "I have no motivation to study",
        "I want to kill myself",
        "give me revision tips",
        "my exam is in 5 days make me a plan",
    ]
    try:
        model = joblib.load(config.INTENT_MODEL_DIR / "intent_pipe.joblib")
        for msg in test_msgs:
            probs = model.predict_proba([msg])[0]
            idx = int(np.argmax(probs))
            tag = model.classes_[idx]
            conf = probs[idx]
            print(f"  '{msg[:50]}'")
            print(f"   → {tag} ({conf*100:.1f}% confidence)")
    except FileNotFoundError:
        print("  (train the model first: python -m src.intents_train)")

    print(f"\n{'='*60}")
    print("  Evaluation complete!")
    print(f"{'='*60}\n")

    return acc, f1


if __name__ == "__main__":
    evaluate_intent_classifier()