import json
from langchain_openai import ChatOpenAI
from src.llm.llm_service import llm
from src.agents.prompts import PLANNER_PROMPT
from src.agents.state import AgentState



def planner(state: AgentState):

    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    response = llm.invoke(
        [
            ("system", PLANNER_PROMPT),
            ("human", query),
        ]
    )

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    result = json.loads(raw)

    return {
        "tools": result["tools"],
        "reason": result["reason"],
    }