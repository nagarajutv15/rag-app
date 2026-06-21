from langchain_core.prompts import ChatPromptTemplate

from src.agents.prompts import ROUTER_PROMPT
from src.llm.llm_service import llm


prompt = ChatPromptTemplate.from_template(
    ROUTER_PROMPT
)

router_chain = prompt | llm


def route_question(question: str) -> str:

    response = router_chain.invoke(
        {
            "question": question
        }
    )

    return response.content.strip().lower()