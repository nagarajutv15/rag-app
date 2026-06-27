from langchain_openai import ChatOpenAI

from src.agents.prompts import GENERATOR_PROMPT
from src.llm.llm_service import llm



def generator(state):

    prompt = GENERATOR_PROMPT.format(
        memory=state.get("memory_context", ""),
        rag=state.get("rag_context", ""),
        web=state.get("web_context", "")
    )

    response = llm.invoke(
        [
            ("system", prompt),
            ("human", state["question"])
        ]
    )

    return {
        "answer": response.content
    }