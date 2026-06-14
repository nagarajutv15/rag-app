from datetime import datetime

from src.models.chat_session import (
    ChatSession
)

from src.models.chat_message import (
    ChatMessage
)


def create_session(db):

    session = ChatSession()

    db.add(session)

    db.commit()

    db.refresh(session)

    return session


def save_message(
    db,
    session_id: str,
    role: str,
    content: str
):

    message = ChatMessage(

        session_id=session_id,

        role=role,

        content=content
    )

    db.add(message)

    session = (

        db.query(ChatSession)

        .filter(
            ChatSession.session_id
            == session_id
        )

        .first()
    )

    if session:

        session.last_activity_at = (
            datetime.utcnow()
        )

    db.commit()


def get_chat_history(
    db,
    session_id: str,
    limit: int = 10
):

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

    return messages[-limit:]