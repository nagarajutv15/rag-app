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
You are a query routing agent.

Possible routes:

HR
FINANCE
IT
LEGAL
WEB

Rules:

HR:
leave, attendance,
promotion, benefits,
remote work

FINANCE:
salary, payroll,
expenses, reimbursement,
budget

IT:
password, MFA,
security, VPN,
software

LEGAL:
privacy, GDPR,
compliance,
confidentiality,
contracts

WEB:
latest news,
current events,
recent changes,
information not likely
stored in company documents

Return only the route.

Question:

{query}
"""

    result = structured_llm.invoke(
        prompt
    )

    return result.route