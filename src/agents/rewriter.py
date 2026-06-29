import time

from src.agents.prompts import REWRITER_PROMPT
from src.agents.state import AgentState
from src.llm.llm_service import llm
from src.utils.logger import logger


async def rewriter(state: AgentState):

    start = time.perf_counter()

    logger.info(
        "Rewriter Started | Retry=%d",
        state["retry_count"] + 1,
    )

    # ------------------------------------------------------------------
    # Decide why we're rewriting
    # ------------------------------------------------------------------

    if not state["retrieval_success"]:

        reason = (
            "No relevant documents were retrieved for the user's question."
        )

    elif state["needs_more_context"]:

        reason = (
            "The generated answer requires additional context."
        )

    else:

        reason = (
            "Improve retrieval quality."
        )

    try:

        prompt = REWRITER_PROMPT.format(
            question=state["question"],
            reason=reason,
        )

        response = await llm.ainvoke(
            [
                ("system", prompt),
            ]
        )

        rewritten_query = response.content.strip()

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Rewriter Completed | Retry=%d | Time=%.2f ms",
            state["retry_count"] + 1,
            latency,
        )

        return {

            "rewritten_question": rewritten_query,

            "retry_count": state["retry_count"] + 1,

            "retry_reason": reason,

            "observability": {

                **state.get(
                    "observability",
                    {},
                ),

                "rewriter": {

                    "reason": reason,

                    "query": rewritten_query,

                    "latency_ms": round(
                        latency,
                        2,
                    ),

                }

            }

        }

    except Exception:

        logger.exception(
            "Rewriter Failed"
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        return {

            "rewritten_question": state.get(
                "rewritten_question",
                state["question"],
            ),

            "retry_count": state["retry_count"] + 1,

            "retry_reason": reason,

            "observability": {

                **state.get(
                    "observability",
                    {},
                ),

                "rewriter": {

                    "reason": reason,

                    "query": state.get(
                        "rewritten_question",
                        state["question"],
                    ),

                    "latency_ms": round(
                        latency,
                        2,
                    ),

                    "status": "failed",

                }

            }

        }