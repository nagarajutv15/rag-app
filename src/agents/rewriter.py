import time

from src.llm.llm_service import llm
from src.agents.prompts import REWRITER_PROMPT
from src.agents.state import AgentState
from src.utils.logger import logger


async def rewriter(state: AgentState):

    start = time.perf_counter()

    logger.info(
        "Rewriter Started | Retry=%d",
        state["retry_count"] + 1,
    )

    prompt = REWRITER_PROMPT.format(
        question=state["question"],
        reason=state["evaluation"]["reason"],
    )

    response = await llm.ainvoke(
        [
            ("system", prompt),
        ]
    )

    rewritten_query = response.content.strip()

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Rewriter Completed | Time=%.2f ms",
        latency,
    )

    return {
        "rewritten_question": rewritten_query,
        "retry_count": state["retry_count"] + 1,
        "observability": {
            **state.get("observability", {}),
            "rewriter": {
                "query": rewritten_query,
                "latency_ms": round(latency, 2),
            }
        }
    }

