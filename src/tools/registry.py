from src.tools.memory_tool import MemoryTool
from src.tools.rag_tool import RAGTool
from src.tools.web_tool import WebTool
from src.tools.llm_tool import LLMTool

TOOL_REGISTRY = {
    "memory": MemoryTool,
    "rag": RAGTool,
    "web": WebTool,
    "llm": LLMTool,
}