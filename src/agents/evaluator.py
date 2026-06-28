import json
import time

from src.llm.llm_service import llm
from src.agents.prompts import EVALUATOR_PROMPT
from src.agents.state import AgentState
from src.utils.logger import logger


async def evaluator(state: AgentState):

    start = time.perf_counter()

    logger.info("Evaluator Started")

    prompt = EVALUATOR_PROMPT.format(
        question=state["question"],
        memory=state.get("memory_context", ""),
        rag=state.get("rag_context", ""),
        web=state.get("web_context", ""),
        answer=state["answer"],
    )

    response = await llm.ainvoke(
        [
            ("system", prompt),
        ]
    )

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    evaluation = json.loads(raw)

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Evaluator Completed | Passed=%s | Time=%.2f ms",
        evaluation["is_answer_sufficient"],
        latency,
    )

    return {
        "evaluation": evaluation,
        "observability": {
            **state.get("observability", {}),
            "evaluation": {
                "passed": evaluation["is_answer_sufficient"],
                "reason": evaluation["reason"],
                "latency_ms": round(latency, 2),
            }
        }
    }

