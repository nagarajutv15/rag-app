from sqlalchemy import text

from src.models import SessionLocal

from src.agents.tools.sql_generator import generate_sql
from src.agents.tools.sql_guard import validate_sql
from src.llm.llm_service import llm

class SQLTool:

    @staticmethod
    def execute(question: str):

        db = SessionLocal()

        try:

            sql = generate_sql(question)

            print(f"Generated SQL:\n{sql}")

            validate_sql(sql)

            result = db.execute(text(sql))

            rows = result.fetchall()

            rows_data = [
                dict(row._mapping)
                for row in rows
            ]

            rows_data = [
                dict(row._mapping)
                for row in rows
            ]
            
            answer = format_sql_answer(
                question,
                rows_data
            )
            
            return {
                "answer": answer,
            }

        finally:
            db.close()





def format_sql_answer(
        question: str,
        rows: list
):

    prompt = f"""
Question:

{question}

Database Result:

{rows}

Generate a concise natural language answer.
"""

    response = llm.invoke(prompt)

    return response.content