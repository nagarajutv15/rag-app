from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# Rerank retrieved documents based on semantic relevance
# between query and document text.

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

    reranked_documents = []

    for doc, score in zip(
        documents,
        scores
    ):

        doc["rerank_score"] = float(
            score
        )

        reranked_documents.append(
            doc
        )

    reranked_documents.sort(

        key=lambda x: x.get(
            "rerank_score",
            0.0
        ),

        reverse=True
    )

    return reranked_documents[:top_k]