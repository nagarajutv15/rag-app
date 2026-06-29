import time

from src.utils.logger import logger


class LLMTool:

    @staticmethod
    def execute(state):

        start = time.perf_counter()

        query = (
            state.get("rewritten_question")
            or state["question"]
        )

        logger.info(
            "LLM Tool Started | Query=%s",
            query,
        )

        try:

            return {

                "context": (
                    "The planner determined that this question can be answered "
                    "using general knowledge."
                ),

                "sources": []

            }

        except Exception:

            logger.exception(
                "LLM Tool Failed | Query=%s",
                query,
            )

            return {

                "context": "",

                "sources": []

            }

        finally:

            latency = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                "LLM Tool Completed | Time=%.2f ms",
                latency,
            )