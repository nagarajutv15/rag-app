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

            documents = hybrid_search(
                query=query,
                top_k=5,
                min_score=0.35,
            )

            latency = (time.perf_counter() - start) * 1000

            logger.info(
                "RAG Tool Completed | Documents=%d | Time=%.2f ms",
                len(documents),
                latency,
            )

            if not documents:

                return {
                    "context": "",
                    "sources": []
                }

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
                ]
            }

        except Exception:

            logger.exception(
                "RAG Tool Failed | Query=%s",
                query,
            )

            raise