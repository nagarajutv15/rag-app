import re

def extract_metadata(text: str):

    email = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone_number = re.findall(
        r"\+?\d[\d\s-]{8,}",
        text
    )

    return {
        "emails": email,
        "phone_numbers": phone_number
    }
