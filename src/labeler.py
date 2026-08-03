"""
labeler.py

Bootstraps suspicious/genuine labels using a rule-based heuristic over
engineered features (features.py), since the dataset has no ground-truth
fake/genuine column.

IMPORTANT DESIGN NOTE (read before touching classifier.py):
The heuristic in assign_label() uses engineered features (promotional
phrase count, repeated word ratio, lexical diversity, uppercase ratio,
exclamation count, review length) to PRODUCE the label. If the downstream
classifier is then trained on those same engineered features, it will
just learn to reconstruct this if/else rule rather than learning anything
new from the text itself -- accuracy will look artificially perfect and
mean nothing.

To avoid this, this script preserves the RAW review text (and rating)
alongside the label, separately from the engineered feature columns.
classifier.py should train on TF-IDF of review_text, NOT on the
engineered feature columns saved here. The engineered features are kept
in this file only so they remain available for trust_score.py's
behavioral_signal(), which is a legitimately separate, independent signal
once the classifier no longer relies on the same columns.

Known limitation (state this in the README): since assign_label() is a
heuristic, the classifier's learning ceiling is bounded by how good this
heuristic is. It can generalize the heuristic's pattern to phrasing the
heuristic doesn't explicitly check, but it cannot discover a notion of
"fakeness" beyond what the heuristic implies. This is a real constraint
of working without ground-truth labels, not a hidden flaw.
"""

from pathlib import Path

import pandas as pd

from .preprocessing import preprocess
from .features import build_feature_dataframe

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def assign_label(row):
    """
    Heuristic suspicion score. Returns 1 (suspicious/likely fake) if at
    least 3 of 6 flags trigger, else 0 (likely genuine).

    NOTE: review_length < 8 flags short reviews as suspicious. This can
    mislabel short-but-genuine reviews ("Great fit, fast shipping.").
    Manually spot-check a sample of label=1 rows before trusting this at
    scale -- see build_labeled_dataset()'s printed sample.
    """
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


def build_labeled_dataset(sample_check_n=15):
    """
    Builds the labeled dataset and, critically, preserves raw review_text
    and rating alongside the label -- these are what classifier.py should
    actually train on (via TF-IDF), not the engineered feature columns
    used to produce the label.

    Saves two views:
      labeled_reviews.csv        -> full feature set + label (for
                                     trust_score.py's behavioral_signal
                                     and for inspection/debugging)
      labeled_reviews_text.csv   -> review_id, product_id, review_text,
                                     rating, label only (for classifier.py
                                     to train on, decoupled from the
                                     engineered features)
    """
    df = preprocess()

    # Preserve raw text before it gets dropped by build_feature_dataframe.
    df = df.copy()
    df["review_text"] = (
        df["review_title"].fillna("").astype(str)
        + " "
        + df["review_content"].fillna("").astype(str)
    )
    text_lookup = df[["review_id", "product_id", "review_text", "rating"]]

    feature_df = build_feature_dataframe(df)
    feature_df["label"] = feature_df.apply(assign_label, axis=1)

    # Full feature + label view, for trust_score.py's behavioral signal.
    full_path = DATA_DIR / "labeled_reviews.csv"
    feature_df.to_csv(full_path, index=False)

    # Text-only view, for classifier.py. Merge back in the raw text.
    text_df = feature_df[["review_id", "product_id", "label"]].merge(
        text_lookup, on=["review_id", "product_id"], how="left"
    )
    text_path = DATA_DIR / "labeled_reviews_text.csv"
    text_df.to_csv(text_path, index=False)

    print(f"Full feature + label dataset saved to {full_path}")
    print(f"Text-only dataset for classifier training saved to {text_path}")
    print()
    print("Label distribution:")
    print(feature_df["label"].value_counts())

    # Manual sanity-check sample: print a few of each label so you can
    # eyeball whether assign_label is flagging genuinely suspicious
    # reviews or just short/blunt ones. Do this before trusting the
    # labels at scale.
    print()
    print(f"--- Sample of {sample_check_n} label=1 (suspicious) reviews ---")
    suspicious_sample = text_df[text_df["label"] == 1].sample(
        min(sample_check_n, (text_df["label"] == 1).sum()),
        random_state=42
    )
    for _, row in suspicious_sample.iterrows():
        print(f"[{row['review_id']}] {row['review_text'][:120]}")

    print()
    print(f"--- Sample of {sample_check_n} label=0 (genuine) reviews ---")
    genuine_sample = text_df[text_df["label"] == 0].sample(
        min(sample_check_n, (text_df["label"] == 0).sum()),
        random_state=42
    )
    for _, row in genuine_sample.iterrows():
        print(f"[{row['review_id']}] {row['review_text'][:120]}")

    return feature_df, text_df


if __name__ == "__main__":
    build_labeled_dataset()
