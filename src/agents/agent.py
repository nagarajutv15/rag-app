from src.agents.graph import graph

from src.services.memory_service import (
    create_session,
    save_message,
)


class Agent:

    @staticmethod
    def execute(
        question: str,
        session_id: str | None,
        db,
    ):

        # ---------------------------------------------------------
        # Create Session if Required
        # ---------------------------------------------------------

        if not session_id:

            session = create_session(db)

            session_id = session.session_id

        # ---------------------------------------------------------
        # Initial Agent State
        # ---------------------------------------------------------

        state = {

            "question": question,

            "rewritten_question": "",

            "session_id": session_id,

            "tools": [],

            "reason": "",

            "memory_context": "",

            "rag_context": "",

            "web_context": "",

            "answer": "",

            "evaluation": {
                "is_answer_sufficient": False,
                "reason": ""
            },

            "retry_count": 0
        }

        # ---------------------------------------------------------
        # Execute Graph
        # ---------------------------------------------------------

        result = graph.invoke(state)

        # ---------------------------------------------------------
        # Save Conversation
        # ---------------------------------------------------------

        save_message(
            db=db,
            session_id=session_id,
            role="user",
            content=question,
        )

        save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=result["answer"],
        )

        # ---------------------------------------------------------
        # Response
        # ---------------------------------------------------------

        return {

            "session_id": session_id,

            "answer": result["answer"]

        }