import time

from src.rag.hybrid_search import hybrid_search
from src.utils.logger import logger


class RAGTool:

    @staticmethod
    def execute(state):

        start = time.perf_counter()

        query = (
            state.get("rewritten_question")
            or state["question"]
        )

        logger.info(
            "RAG Tool Started | Query=%s",
            query,
        )

        try:

            search_result = hybrid_search(
                query=query,
                top_k=5,
                min_score=0.35,
            )

            documents = search_result["documents"]

            latency = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                "RAG Tool Completed | Documents=%d | BestScore=%.3f | Time=%.2f ms",
                search_result["retrieved_docs"],
                search_result["best_rerank_score"],
                latency,
            )

            return {

                "context": "\n\n".join(
                    doc["text"]
                    for doc in documents
                ),

                "sources": [

                    {
                        "type": "document",
                        "file_name": doc["file_name"],
                        "document_id": doc["document_id"],
                        "chunk_id": doc["chunk_id"],
                    }

                    for doc in documents

                ],

                # -----------------------------
                # Retrieval Metadata
                # -----------------------------

                "retrieved_docs": search_result["retrieved_docs"],

                "best_rerank_score": search_result["best_rerank_score"],

                "retrieval_success": search_result["retrieval_success"],

            }

        except Exception:

            logger.exception(
                "RAG Tool Failed | Query=%s",
                query,
            )

            return {

                "context": "",

                "sources": [],

                "retrieved_docs": 0,

                "best_rerank_score": 0.0,

                "retrieval_success": False,

            }