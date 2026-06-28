import time
from src.agents.generator_stream import generator_stream
from src.agents.graph import graph
from src.services.memory_service import (
    create_session,
    save_message,
)
from src.utils.logger import logger
from src.agents.planner import planner
from src.agents.retrieval import retrieval_node


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

            "llm_context": "",

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

    @staticmethod
    async def execute_stream(
        question: str,
        session_id: str | None,
        db,
    ):

        start_time = time.perf_counter()

        logger.info("Streaming Agent Started")

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

            "llm_context": "",

            "answer": "",

            "evaluation": {
                "is_answer_sufficient": True,
                "reason": ""
            },

            "retry_count": 0,

            "observability": {},
        }

        # ---------------------------------------------------------
        # Planner
        # ---------------------------------------------------------

        planner_result = await planner(state)

        state.update(planner_result)

        # ---------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------

        retrieval_result = await retrieval_node(state)

        state.update(retrieval_result)

        # ---------------------------------------------------------
        # Save User Message
        # ---------------------------------------------------------

        save_message(
            db=db,
            session_id=session_id,
            role="user",
            content=question,
        )

        # ---------------------------------------------------------
        # Stream Assistant
        # ---------------------------------------------------------

        answer = ""

        async for token in generator_stream(state):

            answer += token

            yield token

        # ---------------------------------------------------------
        # Save Assistant Message
        # ---------------------------------------------------------

        save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Streaming Agent Completed | Latency=%.2f ms",
            elapsed,
        )