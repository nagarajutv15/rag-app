from src.rag.adaptive.query_classifier import classify_query

from src.rag.prompts.prompts import build_prompt, sanitize_input

from src.llm.llm_service import generate_response

from src.rag.adaptive.web_search import web_search

from src.rag.adaptive.adaptive_rag import adaptive_retrieve

from src.services.memory_service import (
    get_chat_history,
    save_message
)


#----------------------------------------------------------------------------------------------------------#
# This is the function handles the incmoing requests to the RAG system.

def ask_rag_question(
    query: str,
    session_id: str,
    db
):

    query = sanitize_input(query)

    route = classify_query(query)

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



#----------------------------------------------------------------------------------------------------------#
# This function handles queries that are classified as department-specific. 

def handle_department_query(
    query,
    route,
    session_id,
    db,
    conversation_history
):

    retrieved_docs = adaptive_retrieve(
        query=query,
        department_id=route.department
    )

    if not retrieved_docs:

        return {
            "route": route.route,
            "department": route.department,
            "answer": "No relevant documents found."
        }

    context_chunks = [

        doc.get("text", "")

        for doc in retrieved_docs
    ]

    prompt = build_prompt(
        question=query,
        context_chunks=context_chunks,
        conversation_history=conversation_history
    )

    answer = generate_response(prompt)

    save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=answer
    )

    return {
        "session_id": session_id,
        "route": route.route,
        "department": route.department,
        "answer": answer,
        "sources": [
            {
                "file_name": doc.get("file_name"),
                "version": doc.get("version"),
                "chunk_id": doc.get("chunk_id")
            }
            for doc in retrieved_docs
        ]
}


#----------------------------------------------------------------------------------------------------------#
# This function handles queries that are classified as web search queries. 

def handle_web_query(
    query,
    session_id,
    db
):

    web_context = web_search(query)

    prompt = f"""
Use only the web search results below.

Web Results:

{web_context}

Question:

{query}

Answer:
"""

    answer = generate_response(prompt)

    save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=answer
    )

    return {
        "session_id": session_id,
        "route": "WEB",
        "answer": answer
    }


#----------------------------------------------------------------------------------------------------------#
# This function handles general queries that do not fit into specific categories. 

def handle_general_query(
    query,
    session_id,
    db
):

    answer = generate_response(query)

    save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=answer
    )

    return {
        "session_id": session_id,
        "route": "GENERAL",
        "answer": answer
    }