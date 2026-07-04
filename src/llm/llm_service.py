from langsmith import traceable
import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import ChatOpenAI
from src.utils.logger import logger



load_dotenv()


LLM_MODEL     = os.getenv("LLM_MODEL",     "gpt-4o-mini")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4.1-nano")
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.3"))


# ---------------------------------------------------------
# Cached LLM clients (single instance reused across calls)
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOpenAI:
    return ChatOpenAI(model=PLANNER_MODEL, temperature=0.0)

# Convenience singletons used across the codebase
llm         = get_llm()
planner_llm = get_planner_llm()

# ---------------------------------------------------------
# Optional Sync Version
# ---------------------------------------------------------

@traceable(name="Generate Response")
async def generate_response(
    prompt: str,
) -> str:
    
    response = await llm.ainvoke( prompt, ) 
    return response.content


@traceable(name="Generate Response Sync")
def generate_response_sync(
    prompt: str,
) -> str:

    response = llm.invoke(prompt)
    return response.content