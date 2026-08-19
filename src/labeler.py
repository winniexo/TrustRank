"""Generate transparent heuristic labels for the source records.

The Amazon source does not contain verified fake/genuine labels. The labels
here are therefore *bootstrap heuristic labels*, not ground truth.
"""

from pathlib import Path

import pandas as pd

from .preprocessing import preprocess
from .features import build_feature_dataframe

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def assign_label(row):
    """Return 1 for suspicious and 0 for likely genuine using 6 signals."""
    suspicion = 0
    if row["promotional_phrase_count"] > 0:
        suspicion += 1
    if row["repeated_word_ratio"] > 0.20:
        suspicion += 1
    if row["lexical_diversity"] < 0.45:
        suspicion += 1
    if row["uppercase_ratio"] > 0.15:
        suspicion += 1
    if row["exclamation_count"] > 3:
        suspicion += 1
    if row["review_length"] < 8:
        suspicion += 1
    return 1 if suspicion >= 3 else 0


def build_labeled_dataset(sample_check_n=10):
    """Create one clean labeled record per source row.

    No merge is performed on review_id/product_id because review_id is an
    aggregated comma-separated field in this source dataset and is not a
    unique key. record_id is the stable row-level key instead.
    """
    df = preprocess().copy()

    df["review_text"] = (
        df["review_title"] + " " + df["review_content"]
    ).str.strip()

    feature_df = build_feature_dataframe(df)
    if "record_id" not in feature_df.columns:
        feature_df.insert(0, "record_id", df["record_id"].values)
    feature_df["label"] = feature_df.apply(assign_label, axis=1).astype(int)

    # Keep the clean text and identifiers directly aligned by record_id.
    text_df = df[["record_id", "product_id", "review_id", "review_text", "clean_review_text", "rating"]].copy()
    text_df = text_df.merge(
        feature_df[["record_id", "label"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    full_path = DATA_DIR / "labeled_reviews.csv"
    text_path = DATA_DIR / "labeled_reviews_text.csv"
    feature_df.to_csv(full_path, index=False)
    text_df.to_csv(text_path, index=False)

    print(f"\nFull feature + label dataset: {len(feature_df)} rows")
    print(f"Text dataset for classifier: {len(text_df)} rows")
    print("\nLabel distribution:")
    print(text_df["label"].value_counts().sort_index())
    print("\nLabel percentages:")
    print((text_df["label"].value_counts(normalize=True).sort_index() * 100).round(2))

    for label, name in [(1, "suspicious"), (0, "genuine")]:
        sample = text_df[text_df["label"] == label].sample(
            min(sample_check_n, int((text_df["label"] == label).sum())),
            random_state=42,
        )
        print(f"\n--- Sample of label={label} ({name}) ---")
        for _, row in sample.iterrows():
            print(f"[{row['record_id']}] {row['review_text'][:140]}")

    return feature_df, text_df


if __name__ == "__main__":
    build_labeled_dataset()
