from src.agents.router import route_question

from src.agents.tools.postgres_tool import SQLTool
from src.agents.tools.rag_tool import RAGTool
from src.agents.tools.web_tool import WebTool
from src.agents.tools.memory_tool import MemoryTool


def router_node(state):

    source = route_question(
        state["question"]
    )

    return {
        **state,
        "source": source
    }


def sql_node(state):

    result = SQLTool.execute(
        state["question"]
    )

    return {
        **state,
        "tool_result": result,
        "answer": result
    }


def rag_node(state):

    result = RAGTool.execute(
        state["question"]
    )

    return {
        **state,
        "tool_result": result,
        "answer": result
    }


def web_node(state):

    result = WebTool.execute(
        state["question"]
    )

    return {
        **state,
        "tool_result": result,
        "answer": result
    }


def memory_node(state):

    result = MemoryTool.execute(
        state["question"]
    )

    return {
        **state,
        "tool_result": result,
        "answer": result
    }