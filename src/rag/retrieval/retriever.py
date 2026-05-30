from src.rag.processing.embeddings import get_embedding_model
from src.rag.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

# Retrieves relevant document chunks based on the input query and optional filters.

def retrieve_documents(
    query: str,
    department_id: str | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
    active_only: bool = True
):

    embedding_model = get_embedding_model()

    query_vector = embedding_model.embed_query(
        query
    )
    
    search_filter = None

    if active_only:

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="is_active",
                    match=MatchValue(
                        value=True
                    )
                )
            ]
        )

    results = QDRANT_CLIENT.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k
    )

    retrieved_chunks = []

    for result in results:

        if result.score < min_score:
            continue

        retrieved_chunks.append({

            "chunk_id": result.payload.get(
                "chunk_id"
            ),

            "document_id": result.payload.get(
                "document_id"
            ),

            "department_id": result.payload.get(
                "department_id"
            ),

            "file_name": result.payload.get(
                "file_name"
            ),

            "version": result.payload.get(
                "version"
            ),

            "is_active": result.payload.get(
                "is_active"
            ),

            "uploaded_by": result.payload.get(
                "uploaded_by"
            ),

            "uploaded_at": result.payload.get(
                "uploaded_at"
            ),

            "text": result.payload.get(
                "text"
            ),

            "score": result.score
        })


    return retrieved_chunks
