from src.rag.vectorstore.bm25_store import bm25_search
from sentence_transformers import CrossEncoder
from src.rag.ingestion.document_ingestion import get_embedding_model
from src.rag.vectorstore.qdrant_connection import (
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


#----------------------------------------------------------------------------------------------------------#
# Hybrid Search:
# BM25 + Vector Search + CrossEncoder Reranking

def hybrid_search(
    query: str,
    department_id: str | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
):
    
    bm25_results = bm25_search(
        query=query,
        department_id=department_id,
        top_k=top_k
    )

    vector_results = retrieve_documents(
        query=query,
        department_id=department_id,
        top_k=top_k,
        min_score=min_score
    )

    merged_results = {}

    # VECTOR RESULTS
    for doc in vector_results:
        chunk_id = doc.get("chunk_id")
        if not chunk_id:
            continue

        merged_results[chunk_id] = {
            **doc,
            "hybrid_score": doc.get("score", 0.0) * 0.7,
        }

    # BM25 RESULTS
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
                "hybrid_score": doc.get("bm25_score", 0.0) * 0.3,
            }

    final_results = list(merged_results.values())

    final_results.sort(
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True,
    )

    reranked_results = rerank(
        query=query,
        documents=final_results,
        top_k=top_k,
    )

    print("=" * 50)
    print(f"Question: {query}")
    print(f"Department: {department_id}")
    print(f"BM25 Results: {len(bm25_results)}")
    print(f"Vector Results: {len(vector_results)}")
    print("=" * 50)

    return reranked_results



#----------------------------------------------------------------------------------------------------------#
# Rerank retrieved documents based on semantic relevance between query and document text.

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
            doc.get("text", ""),
        )
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    reranked_documents = []
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
        reranked_documents.append(doc)

    reranked_documents.sort(
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )

    return reranked_documents[:top_k]




#----------------------------------------------------------------------------------------------------------#
# This Function return the retrived documents based on the query and department_id. 

def retrieve_documents(
    query: str,
    department_id: str | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
    active_only: bool = True
):

    embedding_model = get_embedding_model()

    query_vector = embedding_model.embed_query(query)

    search_conditions = []

    if active_only:

        search_conditions.append(

            FieldCondition(
                key="is_active",
                match=MatchValue(
                    value=True
                )
            )
        )

    if department_id:

        search_conditions.append(

            FieldCondition(
                key="department_id",
                match=MatchValue(
                    value=department_id
                )
            )
        )

    search_filter = None

    if search_conditions:

        search_filter = Filter(
            must=search_conditions
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

        retrieved_chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "department_id": payload.get("department_id"),
            "file_name": payload.get("file_name"),
            "version": payload.get("version"),
            "is_active": payload.get("is_active") if payload.get("is_active") is not None else None,
            "text": payload.get("text", ""),
            "score": float(result.score) if result.score is not None else 0.0,
        })

    return retrieved_chunks