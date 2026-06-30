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
# Retrieval Configuration
# ----------------------------------------------------------------------------------------------------------

VECTOR_WEIGHT = 0.60

BM25_WEIGHT = 0.40

MIN_VECTOR_SCORE = 0.0

MIN_RERANK_SCORE = -2.5

DEFAULT_TOP_K = 10

DEFAULT_RERANK_TOP_K = 5

# ----------------------------------------------------------------------------------------------------------
# Score Normalization
# ----------------------------------------------------------------------------------------------------------


def normalize_vector_score(score: float) -> float:

    return max(0.0, min(score, 1.0))


def normalize_bm25_score(score: float, max_score: float) -> float:

    if max_score <= 0:
        return 0.0

    return score / max_score


# ----------------------------------------------------------------------------------------------------------
# Hybrid Search
# ----------------------------------------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_VECTOR_SCORE,
):

    start = time.perf_counter()

    try:

        logger.info(
            "Hybrid Search Started | Query=%s",
            query,
        )

        # ---------------------------------------------------------
        # Parallel Retrieval
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

        logger.info(
            "Retrieved | BM25=%d | Vector=%d",
            len(bm25_results),
            len(vector_results),
        )

        # ---------------------------------------------------------
        # BM25 Max Score
        # ---------------------------------------------------------

        max_bm25_score = max(
            (
                doc.get("bm25_score", 0.0)
                for doc in bm25_results
            ),
            default=1.0,
        )

        # ---------------------------------------------------------
        # Merge Results
        # ---------------------------------------------------------

        merged_results = {}

        # ---------------- Vector ----------------

        for doc in vector_results:

            chunk_id = doc.get("chunk_id")

            if not chunk_id:
                continue

            vector_score = normalize_vector_score(
                doc.get("score", 0.0)
            )

            merged_results[chunk_id] = {

                **doc,

                "vector_score": vector_score,

                "bm25_score": 0.0,

                "hybrid_score": vector_score * VECTOR_WEIGHT,

            }

        # ---------------- BM25 ----------------

        for doc in bm25_results:

            chunk_id = doc.get("chunk_id")

            if not chunk_id:
                continue

            bm25_score = normalize_bm25_score(
                doc.get("bm25_score", 0.0),
                max_bm25_score,
            )

            if chunk_id in merged_results:

                merged_results[chunk_id]["bm25_score"] = bm25_score

                merged_results[chunk_id]["hybrid_score"] += (

                    bm25_score * BM25_WEIGHT

                )

            else:

                merged_results[chunk_id] = {

                    **doc,

                    "vector_score": 0.0,

                    "bm25_score": bm25_score,

                    "hybrid_score": bm25_score * BM25_WEIGHT,

                }

        merged_results = sorted(

            merged_results.values(),

            key=lambda x: x["hybrid_score"],

            reverse=True,

        )

        logger.info(
            "Hybrid Merge Completed | Candidates=%d",
            len(merged_results),
        )

        logger.info(
            "Hybrid Weights | Vector=%.2f | BM25=%.2f",
            VECTOR_WEIGHT,
            BM25_WEIGHT,
        )

        # ---------------------------------------------------------
        # CrossEncoder
        # ---------------------------------------------------------

        start = time.perf_counter()

        reranked_documents = asyncio.run(
            rerank(
                query=query,
                documents=merged_results,
                top_k=DEFAULT_RERANK_TOP_K,
            )
        )

        logger.info(
            "CrossEncoder Time=%.2f ms",
            (time.perf_counter() - start) * 1000,
        )
        best_score = max(
            (doc.get("rerank_score", 0.0) for doc in reranked_documents),
            default=0.0,
        )

        logger.info("=" * 80)
        logger.info("Question           : %s", query)
        logger.info("BM25 Results       : %d", len(bm25_results))
        logger.info("Vector Results     : %d", len(vector_results))
        logger.info("Merged Candidates  : %d", len(merged_results))
        logger.info("Final Documents    : %d", len(reranked_documents))
        logger.info("Best Rerank Score  : %.3f", best_score)
        logger.info("=" * 80)

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

async def rerank(
    query: str,
    documents: list,
    top_k: int = DEFAULT_RERANK_TOP_K,
):

    try:

        if not documents:

            logger.info(
                "No documents available for reranking."
            )

            return []

        logger.info(
            "CrossEncoder Started | Candidates=%d",
            len(documents),
        )

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

        reranked_documents = []

        for doc, score in zip(documents, scores):

            score = float(score)

            doc["rerank_score"] = float(score)

            logger.debug(
                "Chunk=%s | Hybrid=%.3f | Vector=%.3f | BM25=%.3f | Cross=%.3f",
                doc.get("chunk_id"),
                doc.get("hybrid_score", 0.0),
                doc.get("vector_score", 0.0),
                doc.get("bm25_score", 0.0),
                score,
            )

        # ------------------------------------------------------
        # Sort by rerank score
        # ------------------------------------------------------

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        best_score = documents[0]["rerank_score"]

        if best_score < MIN_RERANK_SCORE:
            logger.info(
                "CrossEncoder rejected all documents | Best Score=%.3f",
                best_score,
            )
            return []

        dynamic_threshold = max(
            best_score - 2.0,
            MIN_RERANK_SCORE,
        )

        reranked_documents = [

            doc

            for doc in documents

            if doc["rerank_score"] >= dynamic_threshold

        ]

        logger.info(
            "CrossEncoder Completed | Best=%.3f | Threshold=%.3f | Passed=%d | Rejected=%d",
            best_score,
            dynamic_threshold,
            len(reranked_documents),
            len(documents) - len(reranked_documents),
        )

        return reranked_documents[:top_k]

    except Exception:

        logger.exception(
            "CrossEncoder reranking failed."
        )

        return []



# ----------------------------------------------------------------------------------------------------------
# Vector Search
# ----------------------------------------------------------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_VECTOR_SCORE,
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
                        match=MatchValue(
                            value=True,
                        ),
                    )
                ]
            )

        results = QDRANT_CLIENT.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
        )

        logger.info(
            "Qdrant returned %d candidates",
            len(results),
        )

        retrieved_chunks = []

        for result in results:

            payload = result.payload or {}

            score = float(result.score)

            if score < min_score:

                logger.info(
                    "Rejected | Chunk=%s | Score=%.4f",
                    payload.get("chunk_id"),
                    score,
                )

                continue

            retrieved_chunks.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "department_id": payload.get("department_id"),
                    "file_name": payload.get("file_name"),
                    "version": payload.get("version"),
                    "is_active": payload.get("is_active"),
                    "text": payload.get("text", ""),
                    "score": score,
                }
            )

        logger.info(
            "Vector Search Completed | Retrieved=%d | Filtered=%d",
            len(retrieved_chunks),
            len(results) - len(retrieved_chunks),
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