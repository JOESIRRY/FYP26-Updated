"""
Eunoia — Sentiment Risk Classifier Training
=============================================
Trains a sentiment classifier using the real 'dair-ai/emotion' dataset
from HuggingFace — 16,000 labelled examples.

Original labels: sadness, joy, love, anger, fear, surprise
Mapped to risk labels:
  distress → sadness (high distress signal)
  anxious  → fear, anger (anxiety/distress signals)
  neutral  → joy, love, surprise (positive/neutral signals)

Run with:
    python -m src.sentiment_train

This gives Eunoia a second trained ML model separate from
the intent classifier, reducing dependency on the Claude API
for risk assessment.
"""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

import config


# Label mapping from emotion dataset to our risk categories
# This connects a general emotion dataset to the student wellbeing context
# sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5
LABEL_MAP = {
    0: "distress",   # sadness → distress
    1: "neutral",    # joy → neutral/positive
    2: "neutral",    # love → neutral/positive
    3: "anxious",    # anger → anxious/distress
    4: "anxious",    # fear → anxious
    5: "neutral",    # surprise → neutral
}


def load_emotion_dataset():
    """Load and map the HuggingFace emotion dataset."""
    try:
        from datasets import load_dataset
        print("📥 Loading emotion dataset from HuggingFace...")
        ds_train = load_dataset("dair-ai/emotion", split="train")
        ds_test  = load_dataset("dair-ai/emotion", split="test")

        # Apply the label mapping to convert emotions to risk categories
        X_train = [item["text"] for item in ds_train]
        y_train = [LABEL_MAP[item["label"]] for item in ds_train]

        X_test  = [item["text"] for item in ds_test]
        y_test  = [LABEL_MAP[item["label"]] for item in ds_test]

        print(f"✅ Loaded {len(X_train)} train + {len(X_test)} test examples")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"⚠️  Could not load HuggingFace dataset: {e}")
        print("   Falling back to built-in examples...")
        return None, None, None, None


def get_fallback_data():
    """Built-in fallback data if HuggingFace is unavailable."""
    # Small hand-written dataset used if the internet is not available
    data = [
        ("i feel completely hopeless about everything", "distress"),
        ("i have been crying every day this week", "distress"),
        ("i feel worthless and like nothing matters", "distress"),
        ("i feel empty inside and dont know why", "distress"),
        ("i feel like i am a burden to everyone", "distress"),
        ("i have no energy to do anything anymore", "distress"),
        ("i feel so alone and nobody understands me", "distress"),
        ("i feel deeply depressed and cant shake it", "distress"),
        ("i feel like giving up on everything", "distress"),
        ("nothing brings me joy anymore", "distress"),
        ("i feel like a complete failure", "distress"),
        ("i have dark thoughts that scare me", "distress"),
        ("i feel like there is no point in trying", "distress"),
        ("i feel sad all the time for no reason", "distress"),
        ("i have been feeling really low for weeks", "distress"),
        ("i feel mentally broken and exhausted", "distress"),
        ("i feel numb and disconnected from everything", "distress"),
        ("i have lost all hope for my future", "distress"),
        ("i feel so anxious and cannot calm down", "anxious"),
        ("i keep worrying about everything constantly", "anxious"),
        ("i feel panicked and overwhelmed by pressure", "anxious"),
        ("i have constant anxiety that wont go away", "anxious"),
        ("i feel nervous and on edge all the time", "anxious"),
        ("i keep having panic attacks", "anxious"),
        ("i feel stressed about my exams and deadlines", "anxious"),
        ("i worry constantly about failing", "anxious"),
        ("i feel overwhelmed by everything on my plate", "anxious"),
        ("i feel terrified about my upcoming exams", "anxious"),
        ("i cannot stop overthinking everything", "anxious"),
        ("i feel scared about my future", "anxious"),
        ("i get panic attacks when thinking about exams", "anxious"),
        ("i feel tense and cannot relax", "anxious"),
        ("i need help planning my revision schedule", "neutral"),
        ("can you give me some study tips", "neutral"),
        ("how do i improve my time management", "neutral"),
        ("i want to be more productive with my studying", "neutral"),
        ("i am looking for advice on sleep routines", "neutral"),
        ("hello i am a student and need some support", "neutral"),
        ("i need a study plan for my upcoming exams", "neutral"),
        ("i feel okay but want to improve my habits", "neutral"),
        ("i am doing reasonably well but want to do better", "neutral"),
        ("i feel fine but my sleep could be better", "neutral"),
        ("i want to be more organised this semester", "neutral"),
        ("i feel good about my progress this week", "neutral"),
        ("i feel positive about my exams this time", "neutral"),
        ("i feel calm and want to plan my revision", "neutral"),
        ("i had a good week and want to keep it going", "neutral"),
    ]
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    return X[:40], y[:40], X[40:], y[40:]


def train_sentiment_classifier():
    """Train and save the sentiment risk classifier."""
    print("=" * 60)
    print("  Eunoia — Sentiment Classifier Training")
    print("=" * 60)

    # Try to load the real HuggingFace dataset first
    X_train, y_train, X_test, y_test = load_emotion_dataset()

    if X_train is None:
        # Fall back to the built-in examples if download failed
        X_all, y_all = get_fallback_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

    # Show how many examples exist per risk category
    print(f"\n📊 Training distribution:")
    for label, count in sorted(Counter(y_train).items()):
        print(f"  {label:<12} {count:>6} examples")

    # Build the TF-IDF + Logistic Regression pipeline
    # Uses 1-2 ngrams and a larger vocabulary since we have more training data
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=20000,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])

    print("\n🔧 Training classifier...")
    pipe.fit(X_train, y_train)

    # Evaluate on the held-out test set
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n{'─'*60}")
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'─'*60}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Cross-validation on the full combined dataset for a reliable accuracy estimate
    X_all = X_train + X_test
    y_all = y_train + y_test
    cv = cross_val_score(pipe, X_all, y_all, cv=5, scoring="accuracy")
    print(f"🔁 5-Fold CV: {cv.mean():.4f} ± {cv.std():.4f}")

    # Show sample predictions to check the model makes sensible decisions
    print("\n🧪 Sample predictions:")
    samples = [
        "i feel so sad and hopeless about everything",
        "i am really anxious and stressed about my exams",
        "i need help with my revision timetable",
        "i feel happy and motivated today",
        "i have been crying and feel completely lost",
        "i keep worrying and cannot stop overthinking",
    ]
    for s in samples:
        probs = pipe.predict_proba([s])[0]
        pred = pipe.classes_[np.argmax(probs)]
        conf = np.max(probs)
        print(f"  '{s[:50]}'")
        print(f"   → {pred} ({conf*100:.1f}%)")

    # Save the trained model to disk using joblib
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / "sentiment_model.joblib"
    joblib.dump(pipe, path)
    print(f"\n✅ Saved sentiment model to {path}")
    print(f"{'='*60}\n")

    return pipe


if __name__ == "__main__":
    train_sentiment_classifier()