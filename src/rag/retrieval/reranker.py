from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Reranking retrieved documents based on relevance to the query

def rerank(
    query: str,
    documents: list,
    top_k: int = 5
):

    if not documents:
        return []

    pairs = [
        (
            query,
            doc.get("text", "")
        )
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )

    scored_docs = []

    for doc, score in zip(
        documents,
        scores
    ):

        doc["rerank_score"] = float(
            score
        )

        scored_docs.append(
            doc
        )

    scored_docs.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return scored_docs[:top_k]