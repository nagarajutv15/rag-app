from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime
)

from datetime import datetime

from src.models.database import Base


class DocumentMetadata(Base):

    __tablename__ = "document_metadata"

    document_id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    file_name = Column(
        String,
        nullable=False
    )

    source = Column(
        String,
        nullable=False
    )

    department_id = Column(
        String,
        nullable=False
    )

    version = Column(
        Integer,
        default=1
    )

    file_path = Column(
        String
    )

    is_active = Column(
        Boolean,
        default=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )