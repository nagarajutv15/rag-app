from langgraph.graph import StateGraph
from langgraph.graph import END

from src.agents.state import AgentState

from src.agents.nodes import (
    router_node,
    sql_node,
    rag_node,
    web_node,
    memory_node
)

builder = StateGraph(AgentState)

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "sql",
    sql_node
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "web",
    web_node
)

builder.add_node(
    "memory",
    memory_node
)

builder.set_entry_point(
    "router"
)


def route(state):

    return state["source"]


builder.add_conditional_edges(
    "router",
    route,
    {
        "sql": "sql",
        "rag": "rag",
        "web": "web",
        "memory": "memory"
    }
)

builder.add_edge(
    "sql",
    END
)

builder.add_edge(
    "rag",
    END
)

builder.add_edge(
    "web",
    END
)

builder.add_edge(
    "memory",
    END
)

graph = builder.compile()