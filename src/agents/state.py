from typing import TypedDict


class AgentState(TypedDict):

    # ------------------------------------------------------------------
    # User
    # ------------------------------------------------------------------

    question: str
    rewritten_question: str
    session_id: str

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    tools: list[str]
    reason: str

    # ------------------------------------------------------------------
    # Retrieved Context
    # ------------------------------------------------------------------

    # memory_context is set once by Agent (summary + last 3 messages)
    # before the graph runs. It is NOT a plannable/retrieval tool.
    memory_context: str
    rag_context: str
    web_context: str
    llm_context: str

    # ------------------------------------------------------------------
    # Retrieval Metadata
    # ------------------------------------------------------------------

    retrieved_docs: int
    best_rerank_score: float
    retrieval_success: bool

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    answer: str
    needs_more_context: bool
    retry_reason: str

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    retry_count: int

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    observability: dict