from sqlalchemy import (
    Column,
    String,
    DateTime
)
from datetime import datetime
from uuid import uuid4
from src.models.database import Base


class ChatSession(Base):

    __tablename__ = "chat_sessions"

    session_id = Column(
        String,
        primary_key=True,
        default=lambda:str(uuid4())
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    last_activity_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
