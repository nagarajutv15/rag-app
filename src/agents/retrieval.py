import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor

from src.tools.registry import TOOL_REGISTRY
from src.utils.logger import logger


async def retrieval_node(state):

    start = time.perf_counter()

    logger.info(
        "Retrieval Started | Tools=%s",
        state["tools"],
    )

    result = {}

    async def execute(tool_name):

        tool = TOOL_REGISTRY[tool_name]

        if inspect.iscoroutinefunction(tool.execute):

            tool_result = await tool.execute(state)

        else:

            tool_result = await asyncio.to_thread(
                tool.execute,
                state,
            )

        return tool_name, tool_result

    tasks = [

        execute(tool_name)

        for tool_name in state["tools"]

    ]

    for tool_name, tool_result in await asyncio.gather(*tasks):

        result[f"{tool_name}_context"] = tool_result["context"]

        result[f"{tool_name}_sources"] = tool_result["sources"]

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Retrieval Completed | Time=%.2f ms",
        latency,
    )

    return {

        **result,

        "observability": {

            **state.get("observability", {}),

            "retrieval": {

                "memory": bool(result.get("memory_context")),

                "rag": bool(result.get("rag_context")),

                "web": bool(result.get("web_context")),

                "llm": bool(result.get("llm_context")),

                "latency_ms": round(latency, 2),

            }

        }

    }