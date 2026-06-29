import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import CrossEncoder
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from src.ingestion.document_ingestion import get_embedding_model
from src.utils.logger import logger
from src.vectorstore.bm25_store import bm25_search
from src.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION,
)


# ----------------------------------------------------------------------------------------------------------
# CrossEncoder
# ----------------------------------------------------------------------------------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ----------------------------------------------------------------------------------------------------------
# Hybrid Search
# ----------------------------------------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
):

    start = time.perf_counter()

    try:

        # ---------------------------------------------------------
        # Run BM25 and Vector Search in parallel
        # ---------------------------------------------------------

        with ThreadPoolExecutor(max_workers=2) as executor:

            bm25_future = executor.submit(
                bm25_search,
                query=query,
                top_k=top_k,
            )

            vector_future = executor.submit(
                retrieve_documents,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )

            bm25_results = bm25_future.result()

            vector_results = vector_future.result()

        # ---------------------------------------------------------
        # Merge Results
        # ---------------------------------------------------------

        merged_results = {}

        for doc in vector_results:

            chunk_id = doc.get("chunk_id")

            if not chunk_id:
                continue

            merged_results[chunk_id] = {

                **doc,

                "hybrid_score": doc.get("score", 0.0) * 0.7,

            }

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

        merged_results = sorted(

            merged_results.values(),

            key=lambda x: x["hybrid_score"],

            reverse=True,

        )

        logger.info(
            "Hybrid Merge Completed | Total=%d",
            len(merged_results),
        )

        # ---------------------------------------------------------
        # CrossEncoder Reranking
        # ---------------------------------------------------------

        reranked_documents = asyncio.run(

            rerank(

                query=query,

                documents=merged_results,

                top_k=top_k,

            )

        )

        best_score = max(

            (
                doc["rerank_score"]
                for doc in reranked_documents
            ),

            default=0.0,

        )

        logger.info("=" * 70)
        logger.info("Question            : %s", query)
        logger.info("BM25 Results        : %d", len(bm25_results))
        logger.info("Vector Results      : %d", len(vector_results))
        logger.info("Merged Results      : %d", len(merged_results))
        logger.info("Reranked Documents  : %d", len(reranked_documents))
        logger.info("Best Rerank Score   : %.3f", best_score)
        logger.info("=" * 70)

        return {

            "documents": reranked_documents,

            "retrieved_docs": len(reranked_documents),

            "best_rerank_score": best_score,

            "retrieval_success": bool(reranked_documents),

        }

    except Exception:

        logger.exception(
            "Hybrid Search Failed | Query=%s",
            query,
        )

        return {

            "documents": [],

            "retrieved_docs": 0,

            "best_rerank_score": 0.0,

            "retrieval_success": False,

        }

    finally:

        latency = (

            time.perf_counter() - start

        ) * 1000

        logger.info(
            "Hybrid Search Finished | Time=%.2f ms",
            latency,
        )


# ----------------------------------------------------------------------------------------------------------
# CrossEncoder Reranking
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# CrossEncoder Reranking
# ----------------------------------------------------------------------------------------------------------

async def rerank(
    query: str,
    documents: list,
    top_k: int = 5,
):

    try:

        if not documents:

            logger.info(
                "No documents available for reranking."
            )

            return []

        pairs = [

            (
                query,
                doc.get("text", ""),
            )

            for doc in documents

        ]

        scores = await asyncio.to_thread(
            reranker.predict,
            pairs,
        )

        # ---------------------------------------------------------
        # Attach rerank scores
        # ---------------------------------------------------------

        for doc, score in zip(documents, scores):

            doc["rerank_score"] = float(score)

            logger.info(
                "Chunk=%s | Hybrid=%.3f | Rerank=%.3f",
                doc.get("chunk_id"),
                doc.get("hybrid_score", 0.0),
                score,
            )

        # ---------------------------------------------------------
        # Sort by rerank score
        # ---------------------------------------------------------

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        logger.info(
            "Top %d documents selected after reranking.",
            min(len(documents), top_k),
        )

        return documents[:top_k]

    except Exception:

        logger.exception(
            "CrossEncoder reranking failed."
        )

        return []


# ----------------------------------------------------------------------------------------------------------
# Vector Search
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# Vector Search
# ----------------------------------------------------------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    active_only: bool = True,
):

    start = time.perf_counter()

    try:

        logger.info(
            "Vector Search Started | Query=%s",
            query,
        )

        embedding_model = get_embedding_model()

        query_vector = embedding_model.embed_query(query)

        search_filter = None

        if active_only:

            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="is_active",
                        match=MatchValue(value=True),
                    )
                ]
            )

        results = QDRANT_CLIENT.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
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
                    "score": float(result.score),
                }
            )

        logger.info(
            "Vector Search Completed | Retrieved=%d",
            len(retrieved_chunks),
        )

        return retrieved_chunks

    except Exception:

        logger.exception(
            "Vector Search Failed | Query=%s",
            query,
        )

        return []

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Vector Search Finished | Time=%.2f ms",
            latency,
        )