import asyncio
import time

from src.agents.prompts import GENERATOR_PROMPT
from src.llm.llm_service import llm
from src.utils.logger import logger


GENERATION_TIMEOUT = 60


async def generator(state):

    start = time.perf_counter()

    logger.info(
        "Generator Started"
    )

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    prompt = GENERATOR_PROMPT.format(
        question=query,
        tools=", ".join(state["tools"]),
        memory=state.get("memory_context", ""),
        rag=state.get("rag_context", ""),
        web=state.get("web_context", ""),
        llm=state.get("llm_context", ""),
    )

    try:

        response = await asyncio.wait_for(

            llm.ainvoke(
                [
                    ("system", prompt),
                ]
            ),

            timeout=GENERATION_TIMEOUT,

        )

        answer = response.content.strip()

        needs_more_context = (
            "NEED_MORE_CONTEXT" in answer
        )

        if needs_more_context:

            answer = answer.replace(
                "NEED_MORE_CONTEXT",
                "",
            ).strip()

        logger.info("Question=%s", query)
        logger.info(
            "Memory Context Length=%d",
            len(state.get("memory_context", "")),
        )

        logger.info("=" * 80)
        logger.info(
            "RAG Context Length=%d chars",
            len(state.get("rag_context", "")),
        )

        logger.info("=" * 80)
        logger.info(
            "Web Context Length=%d",
            len(state.get("web_context", "")),
        )

        logger.info("=" * 80)
        logger.info(
            "Answer Generated | Characters=%d",
            len(response.content),
        )

        logger.info(
            "Generator Completed | Needs More Context=%s",
            needs_more_context,
        )

        return {

            "answer": answer,

            "needs_more_context": needs_more_context,

            "retry_reason": (

                "Generator requested additional context."

                if needs_more_context

                else ""

            ),

            "observability": {

                **state.get(
                    "observability",
                    {},
                ),

                "generator": {

                    "latency_ms": round(
                        (
                            time.perf_counter()
                            - start
                        )
                        * 1000,
                        2,
                    ),

                    "needs_more_context": needs_more_context,

                },

            },

        }

    except asyncio.TimeoutError:

        logger.exception(
            "Generator Timed Out"
        )

        return {

            "answer": "The request timed out while generating the response.",

            "needs_more_context": False,

            "retry_reason": "Generation timeout.",

        }

    except Exception:

        logger.exception(
            "Generator Failed"
        )

        return {

            "answer": "An unexpected error occurred while generating the response.",

            "needs_more_context": False,

            "retry_reason": "Generator exception.",

        }

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Generator Finished | Time=%.2f ms",
            latency,
        )