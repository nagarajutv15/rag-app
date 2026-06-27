from sqlalchemy import (
    Column,
    String,
    DateTime
)
from datetime import datetime
from uuid import uuid4
from src.models.database import Base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone


class ChatSession(Base):

    __tablename__ = "chat_sessions"

    session_id = Column(
        String,
        primary_key=True,
        default=lambda:str(uuid4())
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    last_activity_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan"
    )