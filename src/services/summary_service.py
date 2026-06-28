from src.llm.llm_service import llm
from src.agents.prompts import SUMMARY_PROMPT


async def summarize_conversation(
    history: str,
):

    response = await llm.ainvoke(
        [
            ("system", SUMMARY_PROMPT),
            ("human", history),
        ]
    )

    return response.content.strip()