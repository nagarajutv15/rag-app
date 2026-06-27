from src.tools.memory_tool import MemoryTool
from src.tools.rag_tool import RAGTool
from src.tools.web_tool import WebTool


TOOL_REGISTRY = {
    "memory": MemoryTool,
    "rag": RAGTool,
    "web": WebTool,
}