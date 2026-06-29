import json
import time
import asyncio
from src.llm.llm_service import llm
from src.agents.prompts import PLANNER_PROMPT
from src.agents.state import AgentState
from src.utils.logger import logger


async def planner(state: AgentState):

    start = time.perf_counter()

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    logger.info(
        "Planner Started | Query=%s",
        query,
    )

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    response = await asyncio.wait_for(
        llm.ainvoke(
            [
                ("system", PLANNER_PROMPT),
                ("human", query),
            ]
        ),
        timeout=30,
    )
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    # ---------------------------------------------------------
    # Debug Raw Response
    # ---------------------------------------------------------

    logger.info(
        "Planner Raw Response:\n%s",
        raw,
    )

    # ---------------------------------------------------------
    # Parse JSON safely
    # ---------------------------------------------------------

    try:

        result = json.loads(raw)

    except json.JSONDecodeError:

        logger.exception(
            "Planner returned invalid JSON."
        )

        result = {}

    except Exception:

        logger.exception(
            "Planner failed unexpectedly."
        )

        result = {}
        logger.exception(
            "Planner returned invalid JSON."
        )

        result = {}

    tools = result.get("tools") or []

    if not isinstance(tools, list):
        tools = []

    reason = reason = result.get(
        "reason"
    ) or "Planner did not provide a reason."

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Planner Completed | Tools=%s | Time=%.2f ms",
        tools,
        latency,
    )

    return {

        "tools": tools,

        "reason": reason,

        "observability": {

            **state.get("observability", {}),

            "planner": {

                "tools": tools,

                "reason": reason,

                "latency_ms": round(
                    latency,
                    2,
                ),

            },

        },

    }