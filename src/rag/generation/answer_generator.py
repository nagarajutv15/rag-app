from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_service import llm
from src.rag.prompts.prompts import (
    ANSWER_GENERATOR_PROMPT
)

prompt = ChatPromptTemplate.from_template(
    ANSWER_GENERATOR_PROMPT
)

chain = prompt | llm


def generate_answer(
    question: str,
    context: str
) -> str:

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    return response.content.strip()