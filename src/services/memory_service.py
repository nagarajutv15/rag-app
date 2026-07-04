import logging
import time
from datetime import datetime, timezone
from langsmith import traceable
from sqlalchemy.exc import SQLAlchemyError

from src.models.chat_session import ChatSession
from src.models.chat_message import ChatMessage
from src.services.summary_service import summarize_conversation


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------#
# Create a new chat session
# ----------------------------------------------------------------------------------------------------------#

def create_session(db):

    start = time.perf_counter()

    try:

        session = ChatSession()

        db.add(session)

        db.commit()

        db.refresh(session)

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Session Created | Session=%s | Time=%.2f ms",
            session.session_id,
            latency,
        )

        return session

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to create chat session."
        )

        raise


# ----------------------------------------------------------------------------------------------------------#
# Save a chat message
# ----------------------------------------------------------------------------------------------------------#

async def save_message(
    db,
    session_id: str,
    role: str,
    content: str,
):

    start = time.perf_counter()

    try:

        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id
            )
            .first()
        )

        if session is None:

            raise ValueError(
                f"Session not found: {session_id}"
            )

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
        )

        db.add(message)

        session.last_activity_at = datetime.now(
            timezone.utc
        )

        db.commit()

        db.refresh(message)

        # NOTE: Conversation summary update used to be triggered here with
        # `await maybe_update_summary(...)`. That blocked the response
        # with an extra LLM call on every 10th message. It has been moved
        # to `schedule_conversation_summary()` (bottom of this file),
        # which the Agent fires as a background task AFTER the response
        # is already on its way back to the user.

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Message Saved | Session=%s | Role=%s | Time=%.2f ms",
            session_id,
            role,
            latency,
        )

        return message

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to save message | Session=%s | Role=%s",
            session_id,
            role,
        )

        raise

# ----------------------------------------------------------------------------------------------------------#
# Get chat history
# ----------------------------------------------------------------------------------------------------------#


def get_chat_history(
    db,
    session_id: str,
    limit: int = 3,
):

    start = time.perf_counter()

    try:

        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id
            )
            .first()
        )

        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id
            )
            .order_by(
                ChatMessage.created_at.desc()
            )
            .limit(limit)
            .all()
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Chat History Retrieved | Session=%s | Messages=%d | Time=%.2f ms",
            session_id,
            len(messages),
            latency,
        )

        return {

            "summary": (
                session.conversation_summary
                if session and session.conversation_summary
                else ""
            ),

            "messages": [

                {
                    "role": message.role,
                    "content": message.content,
                }

                for message in reversed(messages)

            ],

        }

    except Exception:

        logger.exception(
            "Failed to retrieve chat history | Session=%s",
            session_id,
        )

        raise


# ----------------------------------------------------------------------------------------------------------#
# Get lightweight memory context (Summary + Last N messages)
# ----------------------------------------------------------------------------------------------------------#
#
# This replaces the old "memory" tool. Instead of letting the planner LLM
# decide whether to fetch conversation history (which cost extra planner
# tokens + a full retrieval-tool round trip), we ALWAYS build a small,
# fixed-size memory context up front:
#
#     Conversation Summary (~150-250 tokens)   -> rolling summary, refreshed
#                                                  every 10 messages
#   + Last 3 exchanges     (~100-300 tokens)   -> raw recent turns
#   --------------------------------------------
#     Total                 ~2k tokens (worst case), typically much less
#
# instead of dumping the entire conversation (5000+ tokens) into every LLM
# call. As a hard safety net (long messages, oversized summary, etc.) the
# final context string is also capped at MAX_MEMORY_CONTEXT_CHARS.
# ----------------------------------------------------------------------------------------------------------#

MAX_MEMORY_CONTEXT_CHARS = 2500


def get_memory_context(
    db,
    session_id: str,
    limit: int = 3,
) -> str:

    if not session_id:
        return ""

    start = time.perf_counter()

    try:

        history = get_chat_history(
            db=db,
            session_id=session_id,
            limit=limit,
        )

        summary = history["summary"]

        messages = history["messages"]

        if not summary and not messages:
            return ""

        recent_conversation = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )

        if summary:

            context = f"""
Conversation Summary
--------------------------------------------------

{summary}

--------------------------------------------------
Recent Conversation
--------------------------------------------------

{recent_conversation}
"""

        else:

            context = f"""
Recent Conversation
--------------------------------------------------

{recent_conversation}
"""

        latency = (
            time.perf_counter() - start
        ) * 1000

        truncated = len(context) > MAX_MEMORY_CONTEXT_CHARS

        if truncated:

            context = context[:MAX_MEMORY_CONTEXT_CHARS] + "\n...[truncated]"

        logger.info(
            "Memory Context Built | Session=%s | Messages=%d | HasSummary=%s | Chars=%d | Truncated=%s | Time=%.2f ms",
            session_id,
            len(messages),
            bool(summary),
            len(context),
            truncated,
            latency,
        )

        return context

    except Exception:

        logger.exception(
            "Failed to build memory context | Session=%s",
            session_id,
        )

        return ""


# ----------------------------------------------------------------------------------------------------------#
# Delete chat history
# ----------------------------------------------------------------------------------------------------------#

def delete_chat_history(
    db,
    session_id: str,
):

    start = time.perf_counter()

    try:

        deleted = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id
            )
            .delete(
                synchronize_session=False,
            )
        )

        db.commit()

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Chat History Deleted | Session=%s | Messages=%d | Time=%.2f ms",
            session_id,
            deleted,
            latency,
        )

        return deleted > 0

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to delete chat history | Session=%s",
            session_id,
        )

        raise

# ----------------------------------------------------------------------------------------------------------#
# Update conversation summary
# ----------------------------------------------------------------------------------------------------------#

async def update_conversation_summary(
    db,
    session_id: str,
):

    start = time.perf_counter()

    try:

        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id
            )
            .first()
        )

        if session is None:

            logger.warning(
                "Conversation Summary Skipped | Session=%s not found",
                session_id,
            )

            return

        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id
            )
            .order_by(
                ChatMessage.created_at.asc()
            )
            .all()
        )

        if not messages:

            logger.info(
                "Conversation Summary Skipped | No messages | Session=%s",
                session_id,
            )

            return

        history = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        if not history.strip():

            logger.info(
                "Conversation Summary Skipped | Empty history | Session=%s",
                session_id,
            )

            return

        summary = (
            await summarize_conversation(
                history
            )
        ).strip()

        if not summary:

            logger.info(
                "Conversation Summary Skipped | Empty summary | Session=%s",
                session_id,
            )

            return

        session.conversation_summary = summary

        db.commit()

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Conversation Summary Updated | Session=%s | Messages=%d | Length=%d | Time=%.2f ms",
            session_id,
            len(messages),
            len(summary),
            latency,
        )

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to update conversation summary | Session=%s",
            session_id,
        )

        raise


# ----------------------------------------------------------------------------------------------------------#
# Maybe update conversation summary
# ----------------------------------------------------------------------------------------------------------#

async def maybe_update_summary(
    db,
    session_id: str,
):

    try:

        count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id
            )
            .count()
        )

        if count == 0:
            return

        if count % 10 != 0:
            return

        logger.info(
            "Conversation Summary Triggered | Session=%s | Messages=%d",
            session_id,
            count,
        )

        await update_conversation_summary(
            db=db,
            session_id=session_id,
        )

    except Exception:

        logger.exception(
            "Failed while checking conversation summary | Session=%s",
            session_id,
        )

        raise


# ----------------------------------------------------------------------------------------------------------#
# Schedule conversation summary update as a background task
# ----------------------------------------------------------------------------------------------------------#
#
# Summarization is an LLM call. Awaiting it inline (the old behavior)
# added extra latency to every 10th response before it reached the user.
# This helper is meant to be fired with asyncio.create_task(...) AFTER
# the response has already been returned/streamed, so it never blocks
# the user-facing request.
#
# It opens/closes its OWN DB session because the request-scoped session
# (from FastAPI's Depends(get_db)) is closed as soon as the request
# finishes, i.e. before this task gets a chance to run.
# ----------------------------------------------------------------------------------------------------------#
@traceable(name="Conversation Summary")
async def schedule_conversation_summary(
    session_id: str,
):

    from src.models.database import SessionLocal

    db = SessionLocal()

    try:

        await maybe_update_summary(
            db=db,
            session_id=session_id,
        )

    except Exception:

        logger.exception(
            "Background Conversation Summary Failed (Non-Fatal) | Session=%s",
            session_id,
        )

    finally:

        db.close()