"""
TrustRank Trust Scoring Module

This module computes trust scores for reviews by combining
Logistic Regression confidence with behavioral features.
The computed trust scores are then used to re-rank reviews
and estimate product trustworthiness.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config: signal weights (must sum to 1.0 across whatever signals are present)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "classifier_confidence": 0.55,
    "behavioral_signal": 0.30,
}

# How much the re-ranked position should weight relevance vs trust.
# 1.0 = ignore relevance, sort purely by trust. 0.0 = ignore trust.
RERANK_TRUST_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# Step 1: normalize each raw signal into a [0, 1] "looks genuine" score
# ---------------------------------------------------------------------------

def classifier_signal(proba_genuine):
    """Clip and return the classifier's genuine-probability score."""
    return float(np.clip(proba_genuine, 0.0, 1.0))


def behavioral_signal(review_row, similar_pair_ratio_threshold=0.85):
    """Convert behavioral features into a single trust-like score."""
    lexical_diversity = float(review_row.get("lexical_diversity", 0.5))
    repeated_word_ratio = float(review_row.get("repeated_word_ratio", 0.0))
    promo_count = float(review_row.get("promotional_phrase_count", 0.0))
    uppercase_ratio = float(review_row.get("uppercase_ratio", 0.0))
    similar_pair_ratio = float(review_row.get("similar_pair_ratio", 0.0))
    word_count = float(review_row.get("word_count", 0.0))

    score = 1.0

    # Penalize high similarity to other reviews of the same product
    # (copy-paste / review-farm behavior).
    score -= 0.4 * similar_pair_ratio

    # Penalize low lexical diversity (generic, templated language).
    score -= 0.15 * (1.0 - lexical_diversity)

    # Penalize repeated words within the review itself.
    score -= 0.1 * repeated_word_ratio

    # Penalize promotional-phrase stuffing (diminishing returns after 2).
    score -= 0.15 * min(promo_count / 3.0, 1.0)

    # Penalize excessive uppercase ("AMAZING PRODUCT MUST BUY").
    score -= 0.1 * uppercase_ratio

    # Very short reviews (a handful of words) are weak positive evidence
    # either way, but extremely short + generic reviews are a common fake
    # pattern. Mild penalty only below ~5 words.
    if word_count < 5:
        score -= 0.1

    return float(np.clip(score, 0.0, 1.0))



# ---------------------------------------------------------------------------
# Step 2: combine signals into one trust score per review
# ---------------------------------------------------------------------------

def compute_review_trust_score(proba_genuine, review_row,
                                 weights=None):
    """Combine available signals into one trust score for a review."""
    weights = weights or WEIGHTS

    signals = {
        "classifier_confidence": classifier_signal(proba_genuine),
        "behavioral_signal": behavioral_signal(review_row),
    }

    
    present = {k: v for k, v in signals.items() if v is not None}
    active_weights = {k: weights[k] for k in present}
    weight_sum = sum(active_weights.values())

    if weight_sum == 0:
        # Shouldn't happen since classifier_confidence is always present,
        # but guard against a misconfigured weights dict.
        trust_score = present.get("classifier_confidence", 0.5)
    else:
        trust_score = sum(
            present[k] * (active_weights[k] / weight_sum) for k in present
        )

    return {
        "trust_score": float(np.clip(trust_score, 0.0, 1.0)),
        "classifier_confidence": signals["classifier_confidence"],
        "behavioral_signal": signals["behavioral_signal"],
        "label": "Genuine" if trust_score >= 0.5 else "Fake",
    }


def compute_batch_trust_scores(reviews_df, proba_genuine_col="proba_genuine",
                                  weights=None):
    """Apply trust scoring to a DataFrame of reviews."""
    records = []

    for _, row in reviews_df.iterrows():
        result = compute_review_trust_score(
            proba_genuine=row[proba_genuine_col],
            review_row=row,
            weights=weights,
        )
        result["review_id"] = row.get("review_id")
        records.append(result)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Step 3: re-ranking (distinct step, run AFTER initial IR ranking)
# ---------------------------------------------------------------------------

def rerank_reviews(ranked_reviews_df, trust_scores_df,
                    relevance_col="similarity_score",
                    trust_weight=RERANK_TRUST_WEIGHT):
    """Blend relevance and trust scores into a new review order."""
    merged = ranked_reviews_df.merge(
        trust_scores_df[["review_id", "trust_score", "classifier_confidence",
                          "behavioral_signal",  "label"]],
        on="review_id",
        how="left",
    )

    # Normalize relevance to [0, 1] in case cosine scores aren't already
    # bounded that way after index narrowing / partial matches.
    max_rel = merged[relevance_col].max()
    merged["relevance_norm"] = (
        merged[relevance_col] / max_rel if max_rel > 0 else 0.0
    )

    merged["original_rank"] = merged[relevance_col].rank(
        ascending=False, method="first"
    ).astype(int)

    merged["final_score"] = (
        trust_weight * merged["trust_score"]
        + (1 - trust_weight) * merged["relevance_norm"]
    )

    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged["reranked_position"] = merged.index + 1

    return merged


def compute_product_trust_score(trust_scores_df):
    """Aggregate review trust scores into a product-level trust score."""
    if trust_scores_df.empty:
        return {
            "trust_score_pct": None,
            "genuine_count": 0,
            "fake_count": 0,
            "total_reviews": 0,
        }

    confidence_weight = (
        (trust_scores_df["classifier_confidence"] - 0.5).abs() * 2
    ).clip(lower=0.05)  # floor so no review has literally zero say

    weighted_trust = (
        trust_scores_df["trust_score"] * confidence_weight
    ).sum() / confidence_weight.sum()

    genuine_count = int((trust_scores_df["label"] == "Genuine").sum())
    fake_count = int((trust_scores_df["label"] == "Fake").sum())

    return {
        "trust_score_pct": round(float(weighted_trust) * 100, 1),
        "genuine_count": genuine_count,
        "fake_count": fake_count,
        "total_reviews": len(trust_scores_df),
    }


if __name__ == "__main__":
    print("trust_score.py loaded successfully.")
