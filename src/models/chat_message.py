from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from src.models.database import Base


class ChatMessage(Base):

    __tablename__ = "chat_messages"

    message_id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        String,
        ForeignKey(
            "chat_sessions.session_id"
        ),
        index=True
    )

    role = Column(
        String,
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.now(timezone.utc)
    )

    session = relationship(
        "ChatSession",
        back_populates="messages"
    )