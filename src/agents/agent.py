import asyncio
import time
from src.agents.generator_stream import generator_stream
from src.agents.graph import graph,stream_graph
from src.services.memory_service import (
    create_session,
    save_message,
    get_memory_context,
    schedule_conversation_summary,
)
from src.utils.logger import logger
from langsmith import traceable


MEMORY_RECENT_MESSAGES = 3


_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:

    task = asyncio.create_task(coro)

    _background_tasks.add(task)

    task.add_done_callback(_background_tasks.discard)



class Agent:

    @staticmethod
    @traceable(name="Chat Request")
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
        # Default Conversation Memory (Summary + Last 3 Messages)
        # ---------------------------------------------------------
        # Built once, up front, instead of being an on-demand tool the
        # planner LLM has to reason about and the retrieval node has to
        # call. Cheaper and simpler.

        memory_context = get_memory_context(
            db=db,
            session_id=session_id,
            limit=MEMORY_RECENT_MESSAGES,
        )

        # ---------------------------------------------------------
        # Initial State
        # ---------------------------------------------------------

        state = {

            "question": question,
            "rewritten_question": "",
            "session_id": session_id,

            "tools": [],
            "reason": "",

            "memory_context": memory_context,
            "rag_context": "",
            "web_context": "",
            "llm_context": "",

            "retrieved_docs": 0,
            "best_rerank_score": 0.0,
            "retrieval_success": False,

            "answer": "",
            "needs_more_context": False,
            "retry_reason": "",

            "retry_count": 0,

            "observability": {
                "memory": {
                    "included": bool(memory_context),
                    "recent_messages": MEMORY_RECENT_MESSAGES,
                },
            },
        }
        # ---------------------------------------------------------
        # Execute Graph (ASYNC)
        # ---------------------------------------------------------

        result = await graph.ainvoke(state)

        # ---------------------------------------------------------
        # Save Conversation
        # ---------------------------------------------------------

        await save_message(
            db=db,
            session_id=session_id,
            role="user",
            content=question,
        )

        await save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=result["answer"],
        )

        # ---------------------------------------------------------
        # Background Summary Update (fire-and-forget)
        # ---------------------------------------------------------
        # Runs AFTER messages are saved, does not block the response
        # below. Uses its own DB session internally.

        _fire_and_forget(
            schedule_conversation_summary(
                session_id=session_id,
            )
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
    @traceable(name="Chat Request")
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
        # Default Conversation Memory (Summary + Last 3 Messages)
        # ---------------------------------------------------------

        memory_context = get_memory_context(
            db=db,
            session_id=session_id,
            limit=MEMORY_RECENT_MESSAGES,
        )

        # ---------------------------------------------------------
        # Initial State
        # ---------------------------------------------------------

        state = {

            "question": question,
            "rewritten_question": "",
            "session_id": session_id,

            "tools": [],
            "reason": "",

            "memory_context": memory_context,
            "rag_context": "",
            "web_context": "",
            "llm_context": "",

            "retrieved_docs": 0,
            "best_rerank_score": 0.0,
            "retrieval_success": False,

            "answer": "",
            "needs_more_context": False,
            "retry_reason": "",

            "retry_count": 0,

            "observability": {
                "memory": {
                    "included": bool(memory_context),
                    "recent_messages": MEMORY_RECENT_MESSAGES,
                },
            },
        }

        # ---------------------------------------------------------
        # Execute Graph (ASYNC)
        # ---------------------------------------------------------

        result = await stream_graph.ainvoke(state)

        # ---------------------------------------------------------
        # Merge Graph Result Into State
        # ---------------------------------------------------------
        # stream_graph.ainvoke() returns a NEW dict with everything the
        # planner/retrieval nodes computed (tools, rag_context,
        # memory_context, etc). It does NOT mutate the original `state`
        # in place. Generator must use the merged result, or it runs
        # against the empty pre-graph state (tools=[], rag_context="").

        state = {**state, **result}

        # ---------------------------------------------------------
        # Save User Message
        # ---------------------------------------------------------

        await save_message(
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

        await save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # ---------------------------------------------------------
        # Background Summary Update (fire-and-forget)
        # ---------------------------------------------------------

        _fire_and_forget(
            schedule_conversation_summary(
                session_id=session_id,
            )
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Streaming Agent Completed | Latency=%.2f ms",
            elapsed,
        )