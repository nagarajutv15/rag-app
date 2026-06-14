from src.rag.retrieval.hybrid_search import (
    hybrid_search
)

from src.rag.prompts.prompt_guard import (
    sanitize_input
)

from src.rag.prompts.templates import (
    build_prompt
)

from src.llm.llm_service import (
    generate_response
)

from src.services.memory_service import (
    get_chat_history,
    save_message
)


def ask_rag_question(
    query: str,
    department_id: str,
    session_id: str,
    db
):

    query = sanitize_input(
        query
    )
    # LOAD CHAT HISTORY

    history = get_chat_history(
        db=db,
        session_id=session_id
    )

    conversation_history = "\n".join(

        f"{msg.role}: {msg.content}"

        for msg in history
    )

    # SAVE USER MESSAGE

    save_message(
        db=db,
        session_id=session_id,
        role="user",
        content=query
    )

    # RETRIEVE DOCUMENTS

    retrieved_docs = hybrid_search(

        query=query,

        department_id=department_id,

        top_k=3
    )

    if not retrieved_docs:

        return {

            "question": query,

            "answer":
                "I couldn't find information in the documents."
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

    answer = generate_response(
        prompt
    )

    # SAVE ASSISTANT MESSAGE

    save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=answer
    )

    return {

        "session_id":
            session_id,

        "question":
            query,

        "department_id":
            department_id,

        "answer":
            answer,

        "sources": [

            doc.get("chunk_id")

            for doc in retrieved_docs
        ]
    }