from src.utils.logger import logger


MAX_RETRIES = 2

# ------------------------------------------------------------------------------
# After Retrieval
# ------------------------------------------------------------------------------

def should_generate(state):

    """
    Decide whether to generate an answer or rewrite the query.

    Only RAG retrieval failures should trigger query rewriting.
    """

    # ------------------------------------------------------------------
    # No RAG selected
    # ------------------------------------------------------------------

    if "rag" not in state["tools"]:

        logger.info(
            "Routing -> Generator | No RAG retrieval required."
        )

        return "generate"

    # ------------------------------------------------------------------
    # RAG succeeded
    # ------------------------------------------------------------------

    if state["retrieval_success"]:

        logger.info(
            "Routing -> Generator | Retrieved Docs=%d | Best Score=%.3f",
            state["retrieved_docs"],
            state["best_rerank_score"],
        )

        return "generate"

    # ------------------------------------------------------------------
    # Retry exhausted
    # ------------------------------------------------------------------

    if state["retry_count"] >= MAX_RETRIES:

        state["retrieval_failed"] = True

        state["retrieval_reason"] = (
            "No relevant documents found after maximum retries."
        )

        logger.info(
            "Routing -> Generator | Max retries reached with no retrieval."
        )

        return "generate"

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    logger.info(
        "Routing -> Rewriter | No relevant RAG documents retrieved."
    )

    state["retry_reason"] = (
        "No relevant internal documents were retrieved."
    )

    return "rewrite"


# ------------------------------------------------------------------------------
# After Generation
# ------------------------------------------------------------------------------

def should_retry_after_generation(state):

    """
    The generator can request another retrieval by setting:

        needs_more_context = True

    """

    if not state["needs_more_context"]:

        logger.info(
            "Routing -> END"
        )

        return "end"

    if state["retry_count"] >= MAX_RETRIES:

        logger.info(
            "Routing -> END | Maximum retries reached."
        )

        return "end"

    logger.info(
        "Routing -> Rewriter | Generator requested more context."
    )

    return "rewrite"