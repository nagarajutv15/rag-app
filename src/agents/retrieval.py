import asyncio
import inspect
import time

from src.tools.registry import TOOL_REGISTRY
from src.utils.logger import logger


# ----------------------------------------------------------------------------------------------------------
# Retrieval Node
# ----------------------------------------------------------------------------------------------------------

async def retrieval_node(state):

    start = time.perf_counter()

    tools = state.get("tools", [])

    logger.info(
        "Retrieval Started | Tools=%s",
        tools,
    )

    if not tools:

        logger.info(
            "Retrieval Skipped | No tools selected."
        )

        return {
            "retrieved_docs": 0,
            "best_rerank_score": 0.0,
            "retrieval_success": False,
        }

    result = {}

    retrieval_metadata = {

        "retrieved_docs": 0,

        "best_rerank_score": 0.0,

        "retrieval_success": False,

    }

    # ------------------------------------------------------------------
    # Execute Individual Tool
    # ------------------------------------------------------------------

    async def execute(tool_name):

        tool_start = time.perf_counter()

        try:

            tool = TOOL_REGISTRY[tool_name]

            if inspect.iscoroutinefunction(tool.execute):

                tool_result = await tool.execute(state)

            else:

                tool_result = await asyncio.to_thread(
                    tool.execute,
                    state,
                )

            latency = (
                time.perf_counter() - tool_start
            ) * 1000

            logger.info(
                "%s Tool Completed | Time=%.2f ms",
                tool_name.upper(),
                latency,
            )

            return tool_name, tool_result

        except Exception:

            latency = (
                time.perf_counter() - tool_start
            ) * 1000

            logger.exception(
                "%s Tool Failed | Time=%.2f ms",
                tool_name.upper(),
                latency,
            )

            return tool_name, {

                "context": "",

                "sources": [],

                "retrieved_docs": 0,

                "best_rerank_score": 0.0,

                "retrieval_success": False,

            }

    # ------------------------------------------------------------------
    # Run Retrieval Tools
    # ------------------------------------------------------------------

    try:

        responses = await asyncio.gather(

            *[
                execute(tool_name)
                for tool_name in tools
            ],

            return_exceptions=False,

        )

        for tool_name, tool_result in responses:

            result[f"{tool_name}_context"] = tool_result.get(
                "context",
                "",
            )

            result[f"{tool_name}_sources"] = tool_result.get(
                "sources",
                [],
            )

            if tool_name == "rag":

                retrieval_metadata.update({

                    "retrieved_docs": tool_result.get(
                        "retrieved_docs",
                        0,
                    ),

                    "best_rerank_score": tool_result.get(
                        "best_rerank_score",
                        0.0,
                    ),

                    "retrieval_success": tool_result.get(
                        "retrieval_success",
                        False,
                    ),

                })

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Retrieval Completed | Tools=%s | Time=%.2f ms",
            state["tools"],
            latency,
        )

        if "rag" in state["tools"]:

            logger.info(
                "RAG Retrieval | Docs=%d | Success=%s | BestScore=%.2f",
                retrieval_metadata["retrieved_docs"],
                retrieval_metadata["retrieval_success"],
                retrieval_metadata["best_rerank_score"],
            )

        return {

            **result,

            **retrieval_metadata,

            "observability": {

                **state.get("observability", {}),

                "retrieval": {

                    "memory": bool(
                        result.get("memory_context")
                    ),

                    "rag": bool(
                        result.get("rag_context")
                    ),

                    "web": bool(
                        result.get("web_context")
                    ),

                    "llm": bool(
                        result.get("llm_context")
                    ),

                    "retrieved_docs": retrieval_metadata[
                        "retrieved_docs"
                    ],

                    "best_rerank_score": retrieval_metadata[
                        "best_rerank_score"
                    ],

                    "retrieval_success": retrieval_metadata[
                        "retrieval_success"
                    ],

                    "latency_ms": round(
                        latency,
                        2,
                    ),

                }

            },

        }

    except Exception:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.exception(
            "Retrieval Node Failed | Time=%.2f ms",
            latency,
        )

        return {

            "memory_context": "",

            "rag_context": "",

            "web_context": "",

            "llm_context": "",

            "retrieved_docs": 0,

            "best_rerank_score": 0.0,

            "retrieval_success": False,

        }