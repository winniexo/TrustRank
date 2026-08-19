from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "amazon.csv"

TEXT_COLUMNS = [
    "product_name",
    "category",
    "about_product",
    "review_title",
    "review_content",
]


def clean_text(text):
    """Normalize text for ML/IR: lowercase, remove punctuation/stop words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def preprocess():
    """Load and clean the raw Amazon product/review records.

    Important: the source CSV is product-level and several review fields are
    comma-separated aggregates. We therefore keep one row per source record
    instead of treating review_id as a unique review key.
    """
    df = pd.read_csv(CSV_PATH)

    original_rows = len(df)

    # Preserve missing text as empty strings, while keeping numeric columns
    # numeric instead of converting the whole DataFrame to object dtype.
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_columns = [
        "rating",
        "rating_count",
        "discounted_price",
        "actual_price",
        "discount_percentage",
    ]
    for col in numeric_columns:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(r"[^0-9.]+", "", regex=True)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    # Rating is required for the current feature set.
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(0, 5)]

    # Remove exact duplicate source records only. Do NOT deduplicate by
    # review_id: the source stores multiple review IDs in one product row.
    df = df.drop_duplicates().copy()

    # Stable unique key for one source/product record.
    df.insert(0, "record_id", range(len(df)))

    # Cleaned text used by the retrieval/document pipeline.
    df["clean_review_text"] = (
        df["review_title"] + " " + df["review_content"]
    ).map(clean_text)

    # Keep the original weighting used by the search engine while ensuring
    # the text is actually cleaned before TF-IDF.
    df["document"] = (
        df["product_name"].map(clean_text) + " " +
        df["product_name"].map(clean_text) + " " +
        df["product_name"].map(clean_text) + " " +
        df["category"].map(clean_text) + " " +
        df["category"].map(clean_text) + " " +
        df["about_product"].map(clean_text) + " " +
        df["review_title"].map(clean_text) + " " +
        df["review_content"].map(clean_text)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df.reset_index(drop=True)
    df["record_id"] = range(len(df))

    print(f"Raw rows: {original_rows}")
    print(f"Clean rows: {len(df)}")
    print(f"Exact duplicate rows removed: {original_rows - len(df)}")
    print(f"Missing values remaining in review text: {int((df['clean_review_text'] == '').sum())}")

    return df
