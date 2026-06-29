import asyncio
import time

from src.agents.prompts import GENERATOR_PROMPT
from src.llm.llm_service import llm
from src.utils.logger import logger


GENERATION_TIMEOUT = 60


async def generator_stream(state):

    start = time.perf_counter()

    logger.info(
        "Streaming Generator Started"
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

    answer = ""

    try:

        stream = llm.astream(
            [
                ("system", prompt),
            ]
        )

        async for chunk in asyncio.wait_for(
            stream.__aiter__(),
            timeout=GENERATION_TIMEOUT,
        ):

            if not chunk.content:
                continue

            logger.debug(
                "Generated Chunk | Length=%d",
                len(chunk.content),
            )

            answer += chunk.content

            yield chunk.content

    except asyncio.TimeoutError:

        logger.exception(
            "Streaming Generator Timed Out"
        )

        yield "\n\nThe response generation timed out."

    except Exception:

        logger.exception(
            "Streaming Generator Failed"
        )

        yield "\n\nAn unexpected error occurred while generating the response."

    finally:

        needs_more_context = (
            "NEED_MORE_CONTEXT" in answer
        )

        if needs_more_context:

            logger.info(
                "Generator requested more context."
            )

        state["answer"] = answer.replace(
            "NEED_MORE_CONTEXT",
            "",
        ).strip()

        state["needs_more_context"] = needs_more_context

        state["retry_reason"] = (
            "Generator requested additional context."
            if needs_more_context
            else ""
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Streaming Generator Completed | Time=%.2f ms | Characters=%d | NeedsMoreContext=%s",
            latency,
            len(state["answer"]),
            needs_more_context,
        )