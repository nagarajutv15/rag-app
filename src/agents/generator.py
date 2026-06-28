import time

from src.agents.prompts import GENERATOR_PROMPT
from src.llm.llm_service import llm
from src.utils.logger import logger


async def generator(state):

    start = time.perf_counter()

    logger.info("Generator Started")

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

    response = await llm.ainvoke(
        [
            ("system", prompt),
        ]
    )

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Generator Completed | Time=%.2f ms",
        latency,
    )

    return {
        "answer": response.content,
        "observability": {
            **state.get("observability", {}),
            "generator": {
                "latency_ms": round(latency, 2),
            }
        }
    }

