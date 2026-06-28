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

    def execute(tool_name):

        tool = TOOL_REGISTRY[tool_name]

        return (
            tool_name,
            tool.execute(state),
        )

    with ThreadPoolExecutor() as executor:

        futures = [
            executor.submit(
                execute,
                tool_name,
            )
            for tool_name in state["tools"]
        ]

        for future in futures:

            tool_name, tool_result = future.result()

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

                "latency_ms": round(latency, 2),

            }

        }

    }