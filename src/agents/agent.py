import time

from src.agents.graph import graph
from src.services.memory_service import (
    create_session,
    save_message,
)
from src.utils.logger import logger


class Agent:

    @staticmethod
    async def execute(
        question: str,
        session_id: str | None,
        db,
    ):

        start = time.perf_counter()

        logger.info(
            "Agent Started | Question=%s",
            question,
        )

        # ---------------------------------------------------------
        # Create Session
        # ---------------------------------------------------------

        if not session_id:

            session = create_session(db)

            session_id = session.session_id

        # ---------------------------------------------------------
        # Initial State
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

            "retry_count": 0,

            "observability": {},
        }

        # ---------------------------------------------------------
        # Execute Graph (ASYNC)
        # ---------------------------------------------------------

        result = await graph.ainvoke(state)

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

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Agent Completed | Time=%.2f ms",
            latency,
        )

        return {

            "session_id": session_id,

            "answer": result["answer"],

        }