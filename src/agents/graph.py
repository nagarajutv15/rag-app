from langgraph.graph import StateGraph
from langgraph.graph import START, END

from src.agents.state import AgentState
from src.agents.planner import planner
from src.agents.retrieval import retrieval_node
from src.agents.generator import generator
from src.agents.rewriter import rewriter
from src.agents.routing import (
    should_generate,
    should_retry_after_generation,
)
from src.utils.logger import logger


builder = StateGraph(AgentState)

# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

builder.add_node("planner", planner)

builder.add_node("retrieval", retrieval_node)

builder.add_node("generator", generator)

builder.add_node("rewriter", rewriter)

# ---------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------

builder.add_edge(START, "planner")

builder.add_edge("planner", "retrieval")

# ---------------------------------------------------------------------
# After Retrieval
# ---------------------------------------------------------------------

builder.add_conditional_edges(
    "retrieval",
    should_generate,
    {
        "generate": "generator",
        "rewrite": "rewriter",
    },
)

# ---------------------------------------------------------------------
# After Generation
# ---------------------------------------------------------------------

builder.add_conditional_edges(
    "generator",
    should_retry_after_generation,
    {
        "rewrite": "rewriter",
        "end": END,
    },
)

builder.add_edge(
    "rewriter",
    "planner",
)

graph = builder.compile()

logger.info("LangGraph compiled successfully.")