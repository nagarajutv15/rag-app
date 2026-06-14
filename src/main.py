from fastapi import FastAPI

from src.api.routes import router
from src.api.session_routes import router as session_router

# Import models so SQLAlchemy registers them
from src.models.document_schema import DocumentMetadata
from src.models.chat_session import ChatSession
from src.models.chat_message import ChatMessage
from src.models.database import Base, engine

app = FastAPI()

app.include_router(router)
app.include_router(session_router)


Base.metadata.create_all(bind=engine)