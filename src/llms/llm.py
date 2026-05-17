import os
from dotenv import load_dotenv

load_dotenv()

provider = os.getenv("LLM_PROVIDER", "openai")
print("LLM PROVIDER:", provider)

if provider == "local":
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="gemma:2b")

else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

    