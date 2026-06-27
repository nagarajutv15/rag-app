import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


# Set the LLM model to be used for generating responses. The default is "gpt-4o-mini"
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

# Set the temperature for the LLM, which controls the randomness of the output. A lower value makes the output more deterministic, while a higher value increases creativity.
TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.3"
    )
)

# Create a global instance of the LLM for reuse across the application
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE
)


#----------------------------------------------------------------------------------------------------------#
# This function generates a response from the LLM based on the provided prompt.

def generate_response(
    prompt: str
) -> str:

    response = llm.invoke(
        prompt
    )

    return response.content