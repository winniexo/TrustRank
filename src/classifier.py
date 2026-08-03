"""
classifier.py

Trains a Logistic Regression classifier to distinguish suspicious (1)
and genuine (0) reviews.

IMPORTANT DESIGN DECISION
-------------------------
The labels are generated heuristically in labeler.py using engineered
behavioral features.

To avoid information leakage, this classifier DOES NOT train on any of
those engineered features.

Instead it learns only from the raw review text using TF-IDF.

This lets the classifier discover textual patterns that generalize beyond
the handcrafted rules.

Pipeline

review text
      │
      ▼
 TF-IDF Vectorizer
      │
      ▼
 Logistic Regression
      │
      ▼
 P(genuine)

Behavioral features remain completely independent and are used later by
trust.py while computing the Trust Score.
"""

import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_classifier():

    df = pd.read_csv(DATA_DIR / "labeled_reviews_text.csv")

    df = df.fillna("")

    X_text = (
        df["review_text"]
        .astype(str)
        .str.lower()
    )

    y = df["label"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy : {accuracy:.4f}\n")

    print(classification_report(y_test, predictions))

    if accuracy > 0.97:
        print(
            "\nWARNING:\n"
            "Accuracy is unusually high.\n"
            "Verify that the labels are not too deterministic.\n"
        )

    save_pickle(model, BASE_DIR / "model.pkl")
    save_pickle(vectorizer, BASE_DIR / "vectorizer.pkl")

    print("\nModel saved successfully.")


_MODEL = None
_VECTORIZER = None


def _get_model_and_vectorizer():
    """Load model.pkl / vectorizer.pkl once per process and reuse them.

    Previously these were re-read from disk on every predict_probability()
    call, i.e. once per review, on every search. Same artifacts, same
    predictions -- just loaded a single time instead of repeatedly.
    """
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        _MODEL = load_pickle(BASE_DIR / "model.pkl")
        _VECTORIZER = load_pickle(BASE_DIR / "vectorizer.pkl")
    return _MODEL, _VECTORIZER


def predict_probability(review_title, review_content):
    """
    Returns

        P(genuine review)

    using only the review text.
    """

    model, vectorizer = _get_model_and_vectorizer()

    text = f"{review_title} {review_content}".lower()

    X = vectorizer.transform([text])

    probabilities = model.predict_proba(X)[0]

    genuine_index = list(model.classes_).index(0)

    return probabilities[genuine_index]


if __name__ == "__main__":
    train_classifier()