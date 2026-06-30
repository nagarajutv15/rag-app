import logging
import time
from datetime import datetime, timezone

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

        try:

            await maybe_update_summary(
                db=db,
                session_id=session_id,
            )

        except Exception:

            logger.exception(
                "Conversation Summary Update Failed (Non-Fatal) | Session=%s",
                session_id,
            )

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
    limit: int = 5,
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