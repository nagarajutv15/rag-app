import json
import time
from src.llm.llm_service import llm
from src.agents.prompts import PLANNER_PROMPT
from src.agents.state import AgentState
from src.utils.logger import logger


async def planner(state: AgentState):

    start = time.perf_counter()

    logger.info(
        "Planner Started | Question: %s",
        state["question"],
    )

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    response = await llm.ainvoke(
        [
            ("system", PLANNER_PROMPT),
            ("human", query),
        ]
    )

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    result = json.loads(raw)

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Planner Completed | Tools=%s | Time=%.2f ms",
        result["tools"],
        latency,
    )

    return {
        "tools": result["tools"],
        "reason": result["reason"],
        "observability": {
            **state.get("observability", {}),
            "planner": {
                "tools": result["tools"],
                "reason": result["reason"],
                "latency_ms": round(latency, 2),
            },
        },
    }

