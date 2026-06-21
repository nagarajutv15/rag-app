from typing import TypedDict, Optional


class AgentState(TypedDict):

    question: str

    source: Optional[str]

    answer: Optional[str]

    tool_result: Optional[str]