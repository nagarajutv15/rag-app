from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query, documents,top_k=5):

    pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    scored_docs = []

    for doc, score in zip(documents, scores):

        doc["rerank_score"] = float(score)

        scored_docs.append(doc)

    scored_docs.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return scored_docs[:5]