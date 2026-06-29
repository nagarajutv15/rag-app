import logging
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError
from src.models.chat_session import ChatSession
from src.models.chat_message import ChatMessage
from src.services.summary_service import summarize_conversation
import asyncio

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------#
# Create a new chat session
# ----------------------------------------------------------------------------------------------------------#

def create_session(db):

    try:

        session = ChatSession()
        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info("Chat session created successfully. session_id=%s", session.session_id)

        return session

    except SQLAlchemyError:

        db.rollback()
        logger.exception("Failed to create chat session.")
        raise


# ----------------------------------------------------------------------------------------------------------#
# Save a chat message
# ----------------------------------------------------------------------------------------------------------#

async def save_message(
    db,
    session_id: str,
    role: str,
    content: str
):

    try:

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )

        db.add(message)

        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id
            )
            .first()
        )

        if session is None:
            raise ValueError("Session not found")

        if session:
            session.last_activity_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(message)

        await maybe_update_summary(
            db,
            session_id,
        )

        logger.info(
            "Chat message saved. session_id=%s role=%s",
            session_id,
            role
        )

        return message

    except SQLAlchemyError:

        db.rollback()

        logger.exception(
            "Failed to save chat message. session_id=%s",
            session_id
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

        logger.info(
            "Retrieved %d chat messages. session_id=%s",
            len(messages),
            session_id,
        )

        return {

            "summary": (
                session.conversation_summary
                if session and session.conversation_summary
                else ""
            ),

            "messages": [

                {
                    "role": msg.role,
                    "content": msg.content,
                }

                for msg in reversed(messages)

            ]

        }

    except SQLAlchemyError:

        logger.exception(
            "Failed to retrieve chat history. session_id=%s",
            session_id,
        )

        raise


# ----------------------------------------------------------------------------------------------------------#
# Delete chat history for a session
# ----------------------------------------------------------------------------------------------------------#

def delete_chat_history(
    db,
    session_id: str
):

    try:

        deleted = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id
            )
            .delete(
                synchronize_session=False
            )
        )

        db.commit()

        logger.info(
            "Deleted %d chat messages. session_id=%s",
            deleted,
            session_id
        )

        return deleted > 0

    except SQLAlchemyError:

        db.rollback()
        logger.exception(
            "Failed to delete chat history. session_id=%s",
            session_id
        )
        raise



# ----------------------------------------------------------------------------------------------------------#
# update chat history 
# ----------------------------------------------------------------------------------------------------------#




async def update_conversation_summary(
    db,
    session_id: str,
):

    try:

        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id
            )
            .first()
        )

        if session is None:
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

        history = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        

        summary = (await summarize_conversation(history)).strip()

        # Skip empty summaries
        if not summary:
            return

        session.conversation_summary = summary

        db.commit()

        logger.info(
            "Conversation summary updated | Session=%s | Messages=%d | Summary Length=%d",
            session_id,
            len(messages),
            len(summary),
        )

    except Exception:

        db.rollback()

        logger.exception(
            "Failed to update conversation summary. session_id=%s",
            session_id,
        )

# ----------------------------------------------------------------------------------------------------------#
# maybe_update_summary chat history 
# ----------------------------------------------------------------------------------------------------------#



async def maybe_update_summary(
    db,
    session_id: str,
):

    count = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id
        )
        .count()
    )

    if count > 0 and count % 10 == 0:
        
        logger.info(
            "Updating conversation summary. session_id=%s message_count=%d",
            session_id,
            count,
        )
                
        await update_conversation_summary(
            db,
            session_id,
        )
