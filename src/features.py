import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROMOTIONAL_PHRASES = [
    "must buy", "highly recommend", "best product", "awesome product",
    "worth every penny", "five stars", "excellent product",
    "amazing product", "totally worth", "buy this", "perfect product"
]


def _words(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def word_count(text):
    return len(_words(text))


def character_count(text):
    return len(str(text))


def average_word_length(text):
    words = _words(text)
    return np.mean([len(w) for w in words]) if words else 0


def sentence_count(text):
    return len([s for s in re.split(r"[.!?]+", str(text)) if s.strip()])


def uppercase_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0
    return sum(c.isupper() for c in letters) / len(letters)


def exclamation_count(text):
    return str(text).count("!")


def question_count(text):
    return str(text).count("?")


def punctuation_ratio(text):
    text = str(text)
    if not text:
        return 0
    return sum(c in string.punctuation for c in text) / len(text)


def lexical_diversity(text):
    words = _words(text)
    if not words:
        return 0
    return len(set(words)) / len(words)


def repeated_word_ratio(text):
    words = _words(text)
    if not words:
        return 0
    counts = Counter(words)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(words)


def promotional_phrase_count(text):
    text = str(text).lower()
    return sum(p in text for p in PROMOTIONAL_PHRASES)


def extract_review_features(text, rating):
    return {
        "word_count": word_count(text),
        "character_count": character_count(text),
        "average_word_length": average_word_length(text),
        "sentence_count": sentence_count(text),
        "uppercase_ratio": uppercase_ratio(text),
        "exclamation_count": exclamation_count(text),
        "question_count": question_count(text),
        "punctuation_ratio": punctuation_ratio(text),
        "lexical_diversity": lexical_diversity(text),
        "repeated_word_ratio": repeated_word_ratio(text),
        "promotional_phrase_count": promotional_phrase_count(text),
        "review_length": character_count(text),
        "rating": float(rating)
    }


def similarity_metrics(reviews, threshold=0.85):
    reviews = [str(r) for r in reviews if str(r).strip()]

    if len(reviews) < 2:
        return pd.Series({
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "similar_pair_ratio": 0.0
        })

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(reviews)

    sim = cosine_similarity(matrix)

    upper = np.triu_indices_from(sim, k=1)
    values = sim[upper]

    if len(values) == 0:
        return pd.Series({
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "similar_pair_ratio": 0.0
        })

    return pd.Series({
        "mean_similarity": float(values.mean()),
        "max_similarity": float(values.max()),
        "similar_pair_ratio": float((values >= threshold).mean())
    })


def build_review_features(df):
    data = df.copy()

    data["review_text"] = (
        data["review_title"].fillna("").astype(str)
        + " "
        + data["review_content"].fillna("").astype(str)
    )

    review_features = data.apply(
        lambda row: extract_review_features(
            row["review_text"],
            row["rating"]
        ),
        axis=1
    )

    review_features = pd.DataFrame(review_features.tolist())

    review_features.insert(0, "product_id", data["product_id"])
    review_features.insert(1, "review_id", data["review_id"])

    return review_features


def build_product_features(df):
    data = df.copy()

    data["review_text"] = (
        data["review_title"].fillna("").astype(str)
        + " "
        + data["review_content"].fillna("").astype(str)
    )

    grouped = data.groupby("product_id")

    similarity = grouped["review_text"].apply(similarity_metrics)

    similarity = similarity.reset_index()
    similarity = similarity.pivot(
        index="product_id",
        columns="level_1",
        values="review_text"
    )

    product_features = grouped.agg(
        review_count=("review_id", "count"),
        average_rating=("rating", "mean"),
        rating_variance=("rating", "var"),
        average_review_length=(
            "review_text",
            lambda x: np.mean([len(str(i)) for i in x])
        ),
    )

    product_features["rating_variance"] = (
        product_features["rating_variance"].fillna(0)
    )

    product_features = product_features.join(similarity)

    return product_features.reset_index()


def build_feature_dataframe(df):
    review_df = build_review_features(df)
    product_df = build_product_features(df)

    features = review_df.merge(
        product_df,
        on="product_id",
        how="left"
    )

    features["rating_deviation"] = (
        features["rating"] - features["average_rating"]
    ).abs()

    return features


if __name__ == "__main__":
    print("features.py loaded successfully.")
