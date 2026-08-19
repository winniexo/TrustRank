"""Train and evaluate the TrustRank Logistic Regression classifier.

The source dataset has no verified fake/genuine labels, so labels are
bootstrapped by labeler.py. The classifier is trained only on text and is
kept independent from the behavioral features used by trust.py.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_classifier():
    df = pd.read_csv(DATA_DIR / "labeled_reviews_text.csv")
    df = df.dropna(subset=["label"]).copy()

    # Use cleaned review text. Fall back to raw text only if an older dataset
    # is supplied without the clean_review_text column.
    text_col = "clean_review_text" if "clean_review_text" in df.columns else "review_text"
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X_text = df[text_col]
    y = df["label"]

    print("Dataset used for training:", len(df))
    print("Label distribution:")
    print(y.value_counts().sort_index())

    # Split RAW TEXT first. This prevents TF-IDF vocabulary/statistics from
    # seeing the held-out test set.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )

    # Fit ONLY on training text; transform test text with the fitted vectorizer.
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Balanced class weights prevent the majority class from dominating the
    # decision boundary. We report recall/F1 for suspicious reviews as key
    # metrics, not accuracy alone.
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Suspicious precision: {precision * 100:.2f}%")
    print(f"Suspicious recall: {recall * 100:.2f}%")
    print(f"Suspicious F1-score: {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Genuine", "Suspicious"],
        zero_division=0,
    ))
    print("Confusion Matrix [rows=actual, cols=predicted]:")
    print(cm)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train_text)),
        "test_rows": int(len(X_test_text)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "tfidf_fit_on": "training_text_only",
        "class_weight": "balanced",
        "accuracy": float(accuracy),
        "accuracy_percent": round(float(accuracy * 100), 2),
        "suspicious_precision": float(precision),
        "suspicious_recall": float(recall),
        "suspicious_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "test_class_distribution": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "label_source": "heuristic bootstrap labels; not verified ground truth",
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        cm,
        index=["Actual Genuine", "Actual Suspicious"],
        columns=["Predicted Genuine", "Predicted Suspicious"],
    ).to_csv(EVAL_DIR / "confusion_matrix.csv")

    save_pickle(model, MODEL_DIR / "model.pkl")
    save_pickle(vectorizer, MODEL_DIR / "vectorizer.pkl")

    print("\nEvaluation saved to evaluation/metrics.json")
    print("Confusion matrix saved to evaluation/confusion_matrix.csv")
    print("Model and vectorizer saved successfully.")

    return metrics


_MODEL = None
_VECTORIZER = None


def _get_model_and_vectorizer():
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        _MODEL = load_pickle(MODEL_DIR / "model.pkl")
        _VECTORIZER = load_pickle(MODEL_DIR / "vectorizer.pkl")
    return _MODEL, _VECTORIZER


def predict_probability(review_title, review_content):
    """Return P(genuine) using only review text."""
    model, vectorizer = _get_model_and_vectorizer()
    text = f"{review_title} {review_content}".lower()
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    genuine_index = list(model.classes_).index(0)
    return probabilities[genuine_index]


if __name__ == "__main__":
    train_classifier()
