import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.3"
    )
)


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE
)


def generate_response(
    prompt: str
) -> str:

    response = llm.invoke(
        prompt
    )

    return response.content