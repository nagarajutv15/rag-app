from langgraph.graph import StateGraph
from langgraph.graph import START, END

from src.agents.state import AgentState
from src.agents.planner import planner
from src.agents.retrieval import retrieval_node
from src.agents.generator import generator
from src.agents.evaluator import evaluator
from src.agents.rewriter import rewriter
from src.agents.routing import should_retry
from src.utils.logger import logger


builder = StateGraph(AgentState)

# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

builder.add_node("planner", planner)

builder.add_node("retrieval", retrieval_node)

builder.add_node("generator", generator)

builder.add_node("evaluator", evaluator)

builder.add_node("rewriter", rewriter)

# ---------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------

builder.add_edge(START, "planner")

builder.add_edge("planner", "retrieval")

builder.add_edge("retrieval", "generator")

builder.add_edge("generator", "evaluator")

# ---------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------

builder.add_conditional_edges(
    "evaluator",
    should_retry,
    {
        "rewrite": "rewriter",
        "end": END,
    },
)

builder.add_edge("rewriter", "planner")

graph = builder.compile()

logger.info("LangGraph compiled successfully.")