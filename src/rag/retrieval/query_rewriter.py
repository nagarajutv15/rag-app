from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_service import llm

from src.rag.prompts.prompts import QUERY_REWRITER_PROMPT



prompt = ChatPromptTemplate.from_template(
    QUERY_REWRITER_PROMPT
)

chain = prompt | llm


def rewrite_query(
    question: str
) -> str:
    """
    Rewrite the user's question to improve document retrieval.

    This function does NOT answer the question.
    It only reformulates the query using terminology that is
    more likely to match enterprise documents.

    Args:
        question: Original user question.

    Returns:
        Rewritten query string.
    """

    response = chain.invoke(
        {
            "question": question
        }
    )

    rewritten_query = response.content.strip()

    print(f"Original Query  : {question}")
    print(f"Rewritten Query : {rewritten_query}")

    return rewritten_query