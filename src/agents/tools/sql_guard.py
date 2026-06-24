FORBIDDEN_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create"
]


def validate_sql(sql: str):

    sql_lower = sql.lower()

    if not sql_lower.strip().startswith("select"):
        raise Exception(
            "Only SELECT queries allowed"
        )

    for keyword in FORBIDDEN_KEYWORDS:

        if keyword in sql_lower:

            raise Exception(
                f"Unsafe SQL detected: {keyword}"
            )