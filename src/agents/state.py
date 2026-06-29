from typing import TypedDict

class EvaluationResult(TypedDict):
    is_answer_sufficient: bool
    reason: str


class AgentState(TypedDict):

    question: str

    rewritten_question: str

    session_id: str

    tools: list[str]

    reason: str

    memory_context: str

    rag_context: str

    web_context: str

    llm_context: str

    answer: str

    evaluation: EvaluationResult

    retry_count: int

    observability: dict