from src.utils.logger import logger


MAX_RETRIES = 2

# ------------------------------------------------------------------------------
# After Retrieval
# ------------------------------------------------------------------------------

def should_generate(state):

    """
    Decide whether to generate an answer or rewrite the query.

    Returns:
        - "generate"
        - "rewrite"
    """

    if state["retrieval_success"]:

        logger.info(
            "Routing -> Generator | Retrieved Docs=%d | Best Score=%.3f",
            state["retrieved_docs"],
            state["best_rerank_score"],
        )

        return "generate"

    if state["retry_count"] >= MAX_RETRIES:

        logger.info(
            "Routing -> Generator | Max retries reached with no retrieval."
        )

        return "generate"

    logger.info(
        "Routing -> Rewriter | No relevant documents retrieved."
    )

    state["retry_reason"] = (
        "No relevant documents were retrieved."
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