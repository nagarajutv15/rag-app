from src.rag.retrieval.reranker import rerank
from src.rag.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)

from src.rag.processing.embeddings import (
    get_embedding_model
)


def retrieve_documents(
    query: str,
    top_k: int = 10,
    min_score: float = 0.3
):

    embedding_model = get_embedding_model()

    query_vector = embedding_model.embed_query(
        query
    )

    results = QDRANT_CLIENT.search(

        collection_name=QDRANT_COLLECTION,

        query_vector=query_vector,

        limit=top_k
    )

    retrieved_chunks = []

    for result in results:

        if result.score < min_score:
            continue

        retrieved_chunks.append({

            "text": result.payload.get(
                "text"
            ),

            "chunk_id": result.payload.get(
                "chunk_id"
            ),

            "score": result.score
        })

    reranked_docs = rerank(
        query=query, 
        documents=retrieved_chunks, 
        top_k=top_k
    )

    return reranked_docs