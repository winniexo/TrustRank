from src.preprocessing import preprocess
from src.filter import invertedindex
from src.retrieval import retrievalengine


df = preprocess()

index = invertedindex(df)

engine = retrievalengine(df)


query = "lenovo laptop"

print(f"\nQuery: {query}\n")

candidates = index.search(query)

print(f"Candidates Found: {len(candidates)}")

results = engine.rank(query, candidates)

print("\nTop Results\n")

print(
    results[
        [
            "product_name",
            "category",
            "similarity_score"
        ]
    ].head(10)
)