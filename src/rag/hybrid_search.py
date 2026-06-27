from concurrent.futures import ThreadPoolExecutor

from src.vectorstore.bm25_store import bm25_search
from sentence_transformers import CrossEncoder
from src.ingestion.document_ingestion import get_embedding_model
from src.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ----------------------------------------------------------------------------------------------------------#
# Hybrid Search:
# BM25 + Vector Search (Parallel) + CrossEncoder Reranking

def hybrid_search(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
):

    # Run BM25 and Vector Search in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:

        bm25_future = executor.submit(
            bm25_search,
            query=query,
            top_k=top_k
        )

        vector_future = executor.submit(
            retrieve_documents,
            query=query,
            top_k=top_k,
            min_score=min_score
        )

        bm25_results = bm25_future.result()
        vector_results = vector_future.result()

    merged_results = {}

    # Merge Vector Results
    for doc in vector_results:

        chunk_id = doc.get("chunk_id")

        if not chunk_id:
            continue

        merged_results[chunk_id] = {
            **doc,
            "hybrid_score": doc.get("score", 0.0) * 0.7
        }

    # Merge BM25 Results
    for doc in bm25_results:

        chunk_id = doc.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in merged_results:

            merged_results[chunk_id]["hybrid_score"] += (
                doc.get("bm25_score", 0.0) * 0.3
            )

        else:

            merged_results[chunk_id] = {
                **doc,
                "hybrid_score": doc.get("bm25_score", 0.0) * 0.3
            }

    final_results = sorted(
        merged_results.values(),
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True
    )

    reranked_results = rerank(
        query=query,
        documents=final_results,
        top_k=top_k
    )

    print("=" * 60)
    print(f"Question       : {query}")
    print(f"BM25 Results   : {len(bm25_results)}")
    print(f"Vector Results : {len(vector_results)}")
    print("=" * 60)

    return reranked_results


# ----------------------------------------------------------------------------------------------------------#
# CrossEncoder Reranking

def rerank(
    query: str,
    documents: list,
    top_k: int = 5,
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

    scores = reranker.predict(pairs)

    reranked_documents = []

    for doc, score in zip(documents, scores):

        doc["rerank_score"] = float(score)
        reranked_documents.append(doc)

    reranked_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_documents[:top_k]


# ----------------------------------------------------------------------------------------------------------#
# Vector Search

def retrieve_documents(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    active_only: bool = True,
):

    embedding_model = get_embedding_model()

    query_vector = embedding_model.embed_query(query)

    search_filter = None

    if active_only:

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="is_active",
                    match=MatchValue(value=True)
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

        payload = result.payload or {}

        retrieved_chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "department_id": payload.get("department_id"),
                "file_name": payload.get("file_name"),
                "version": payload.get("version"),
                "is_active": payload.get("is_active"),
                "text": payload.get("text", ""),
                "score": float(result.score)
            }
        )

    return retrieved_chunks