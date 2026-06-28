import time

from src.llm.llm_service import llm
from src.agents.prompts import GENERATOR_PROMPT
from src.utils.logger import logger


async def generator_stream(state):

    start_time = time.perf_counter()

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    prompt = GENERATOR_PROMPT.format(
        question=query,
        memory=state.get("memory_context", ""),
        rag=state.get("rag_context", ""),
        web=state.get("web_context", ""),
    )

    logger.info("Streaming Generator Started")

    answer = ""

    async for chunk in llm.astream(
        [
            ("system", prompt),
        ]
    ):

        if chunk.content:

            answer += chunk.content

            yield chunk.content

    elapsed = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Streaming Generator Completed | Latency=%.2f ms | Characters=%d",
        elapsed,
        len(answer),
    )