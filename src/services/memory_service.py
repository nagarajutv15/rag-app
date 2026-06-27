import logging
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError
from src.models.chat_session import ChatSession
from src.models.chat_message import ChatMessage


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

def save_message(
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
    limit: int = 10
):

    try:

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
            session_id
        )

        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in reversed(messages)
        ]
    
    except SQLAlchemyError:

        logger.exception(
            "Failed to retrieve chat history. session_id=%s",
            session_id
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