import time

from src.agents.prompts import SUMMARY_PROMPT
from src.llm.llm_service import llm
from src.utils.logger import logger


async def summarize_conversation(
    history: str,
):

    start = time.perf_counter()

    try:

        response = await llm.ainvoke(
            [
                ("system", SUMMARY_PROMPT),
                ("human", history),
            ]
        )

        summary = response.content.strip()

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Conversation Summarized | Characters=%d | Time=%.2f ms",
            len(summary),
            latency,
        )

        return summary

    except Exception:

        logger.exception(
            "Conversation Summary Failed"
        )

        raise