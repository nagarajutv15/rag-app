import json

from langchain_openai import ChatOpenAI

from src.agents.prompts import EVALUATOR_PROMPT
from src.agents.state import AgentState


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def evaluator(state: AgentState):

    prompt = EVALUATOR_PROMPT.format(
        question=state["question"],
        memory=state.get("memory_context", ""),
        rag=state.get("rag_context", ""),
        web=state.get("web_context", ""),
        answer=state["answer"],
    )

    response = llm.invoke(
        [
            ("system", prompt),
        ]
    )

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    evaluation = json.loads(raw)

    return {
        "evaluation": evaluation
    }