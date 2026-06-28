import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


# ---------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini",
)

TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.3",
    )
)

# ---------------------------------------------------------
# Global LLM Instance
# ---------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
)


# ---------------------------------------------------------
# Async Response Generator
# ---------------------------------------------------------

async def generate_response(
    prompt: str,
) -> str:

    response = await llm.ainvoke(
        prompt,
    )

    return response.content


# ---------------------------------------------------------
# Optional Sync Version
# ---------------------------------------------------------

def generate_response_sync(
    prompt: str,
) -> str:

    response = llm.invoke(
        prompt,
    )

    return response.content