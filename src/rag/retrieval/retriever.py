from src.rag.vectorstore.qdrant_client import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)

from src.rag.processing.embeddings import (
    get_embedding_model
)


def retrieve_documents(
    query: str,
    top_k: int = 5,
    min_score: float = 0.5
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

    return retrieved_chunks