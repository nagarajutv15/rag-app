from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_service import llm
from src.rag.prompts.prompts import (
    RETRIEVAL_GRADER_PROMPT
)

prompt = ChatPromptTemplate.from_template(
    RETRIEVAL_GRADER_PROMPT
)

chain = prompt | llm


def grade_retrieval(
    question: str,
    context: str
) -> bool:

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    decision = (
        response.content
        .strip()
        .upper()
    )

    print(f"Retrieval Grade : {decision}")

    return decision == "YES"