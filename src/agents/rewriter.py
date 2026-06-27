from src.llm.llm_service import llm
from src.agents.prompts import REWRITER_PROMPT
from src.agents.state import AgentState



def rewriter(state: AgentState):

    prompt = REWRITER_PROMPT.format(
        question=state["question"],
        reason=state["evaluation"]["reason"],
    )

    response = llm.invoke(
        [
            ("system", prompt),
        ]
    )

    return {
        "rewritten_question": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }