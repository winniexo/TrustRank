from .preprocessing import preprocess
from .filter import invertedindex
from .retrieval import retrievalengine

from .classifier import predict_probability
from .trust import (
    compute_batch_trust_scores,
    rerank_reviews,
)

from .features import build_feature_dataframe


df = preprocess()

feature_df = build_feature_dataframe(df).drop(columns=["rating"], errors="ignore")

index = invertedindex(df)

engine = retrievalengine(df)


def search(query):

    candidates = index.search(query)

    if not candidates:
        return df.iloc[0:0].copy()

    results = engine.rank(query, candidates)

    merged = results.merge(
        feature_df,
        on=["record_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )

    merged["proba_genuine"] = merged.apply(
        lambda row: predict_probability(
            row["review_title"],
            row["review_content"],
        ),
        axis=1,
    )

    trust_scores = compute_batch_trust_scores(
        merged,
        proba_genuine_col="proba_genuine",
    )

    final_results = rerank_reviews(
        merged,
        trust_scores,
        relevance_col="similarity_score",
    )

    return final_results