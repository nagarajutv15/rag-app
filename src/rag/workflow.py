import logging

from src.rag.retrieval.hybrid_search import hybrid_search

from src.rag.retrieval.retrieval_grader import grade_retrieval

from src.rag.retrieval.query_rewriter import rewrite_query

from src.rag.generation.answer_generator import generate_answer

logger = logging.getLogger(__name__)


class RAGWorkflow:

    @staticmethod
    def execute(question: str):

        logger.info(
            "Starting Hybrid Retrieval"
        )

        documents = hybrid_search(
            question
        )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        if grade_retrieval(
            question,
            context
        ):

            logger.info(
                "Retrieved context is sufficient."
            )

            answer = generate_answer(
                question,
                context
            )

            return {
                "answer": answer,
                "documents": documents,
                "rewritten": False,
                "attempts": 1
            }

        logger.info(
            "Retrieved context is insufficient. Rewriting query."
        )

        rewritten_query = rewrite_query(
            question
        )

        logger.info(
            "Searching with rewritten query."
        )

        documents = hybrid_search(
            rewritten_query
        )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        if grade_retrieval(
            question,
            context
        ):

            answer = generate_answer(
                question,
                context
            )

            return {
                "answer": answer,
                "documents": documents,
                "rewritten": True,
                "rewritten_query": rewritten_query,
                "attempts": 2
            }

        logger.warning(
            "Unable to retrieve sufficient context."
        )

        return {
            "answer": (
                "I couldn't find sufficient information "
                "in the uploaded documents."
            ),
            "documents": [],
            "rewritten": True,
            "rewritten_query": rewritten_query,
            "attempts": 2
        }