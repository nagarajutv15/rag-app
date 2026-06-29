from src.utils.logger import logger


class LLMTool:

    @staticmethod
    def execute(state):

        logger.info(
            "LLM Tool Selected | Query=%s",
            state.get("rewritten_question") or state["question"],
        )

        return {
            "context": (
                "The planner determined that this question can be answered "
                "using general knowledge."
            ),
            "sources": []
        }