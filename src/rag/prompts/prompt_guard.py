import re


def sanitize_input(
    text: str
):

    if not text:
        return ""

    dangerous_patterns = [

        r"ignore previous instructions",

        r"ignore all instructions",

        r"system prompt",

        r"reveal prompt",

        r"you are now",

        r"act as",

        r"developer instructions",

        r"assistant instructions"
    ]

    cleaned = text

    for pattern in dangerous_patterns:

        cleaned = re.sub(
            pattern,
            "[FILTERED]",
            cleaned,
            flags=re.IGNORECASE
        )

    return cleaned