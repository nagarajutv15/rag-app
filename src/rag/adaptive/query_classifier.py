from src.models.route_identifier import (
    RouteIdentifier
)

from src.llm.llm_service import llm


def classify_query(
    query: str
):


    structured_llm = (
        llm.with_structured_output(
            RouteIdentifier
        )
    )

    prompt = f"""
You are an enterprise query routing agent.

Possible route values:

DEPARTMENT
WEB
GENERAL

Department values:

HR
FINANCE
IT
LEGAL

Rules:

If the question is about company policies,
documents, procedures, leave,
salary, security, compliance,
return:

route = DEPARTMENT

and choose the correct department.

If the question requires current,
recent, latest, public internet
information:

route = WEB

If the question is common knowledge:

route = GENERAL

Question:

{query}
"""

    result = structured_llm.invoke(prompt)

    return result