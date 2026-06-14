from src.rag.adaptive.query_classifier import (
    classify_query
)

from src.rag.prompts.prompt_guard import (
    sanitize_input
)

from src.services.memory_service import (
    get_chat_history,
    save_message
)

from src.services.query_handlers import (
    handle_department_query,
    handle_web_query,
    handle_general_query
)


def ask_rag_question(
    query: str,
    session_id: str,
    db
):

    query = sanitize_input(
        query
    )

    route = classify_query(
        query
    )

    print("=" * 50)
    print("ROUTE:", route.route)
    print("DEPARTMENT:", route.department)
    print("=" * 50)

    save_message(
        db=db,
        session_id=session_id,
        role="user",
        content=query
    )

    history = get_chat_history(
        db=db,
        session_id=session_id
    )

    conversation_history = "\n".join(
        f"{msg.role}: {msg.content}"
        for msg in history
    )

    if route.route == "DEPARTMENT":

        return handle_department_query(
            query=query,
            route=route,
            session_id=session_id,
            db=db,
            conversation_history=conversation_history
        )

    if route.route == "WEB":

        return handle_web_query(
            query=query,
            session_id=session_id,
            db=db
        )

    return handle_general_query(
        query=query,
        session_id=session_id,
        db=db
    )