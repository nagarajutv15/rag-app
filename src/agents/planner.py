import json
import time
import asyncio
from langsmith import traceable
from src.llm.llm_service import planner_llm
from src.agents.prompts import PLANNER_PROMPT
from src.agents.state import AgentState
from src.utils.logger import logger


@traceable(name="Planner")
async def planner(state: AgentState):

    start = time.perf_counter()

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    method = "llm"

    try:

        response = await asyncio.wait_for(
            planner_llm.ainvoke(
                [
                    ("system", PLANNER_PROMPT),
                    ("human", query),
                ]
            ),
            timeout=30,
        )

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
        tools = result.get("tools") or []

    except Exception:

        logger.exception("Planner LLM failed.")
        tools = ["llm"]

    if not isinstance(tools, list):
        tools = []

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Planner Completed | Method=%s | Tools=%s | Time=%.2f ms",
        method, tools, latency,
    )

    return {
        "tools": tools,
        "observability": {
            **state.get("observability", {}),
            "planner": {
                "tools":      tools,
                "method":     method,
                "latency_ms": round(latency, 2),
            },
        },
    }
