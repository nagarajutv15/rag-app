from langchain_core.prompts import ChatPromptTemplate

from src.agents.tools.schema import (
    DATABASE_SCHEMA,
    SQL_GENERATION_PROMPT
)

from src.llm.llm_service import llm

prompt = ChatPromptTemplate.from_template(
    SQL_GENERATION_PROMPT
)

chain = prompt | llm


def generate_sql(question: str):

    response = chain.invoke(
        {
            "schema": DATABASE_SCHEMA,
            "question": question
        }
    )

    sql = response.content.strip()

    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")
    sql = sql.strip()

    return sql