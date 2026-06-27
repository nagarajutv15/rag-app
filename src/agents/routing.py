MAX_RETRIES = 3


def should_retry(state):

    if state["evaluation"]["is_answer_sufficient"]:
        return "end"

    if state["retry_count"] >= MAX_RETRIES:
        return "end"

    return "rewrite"