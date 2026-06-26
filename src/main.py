from fastapi import FastAPI
from src.api.routes import router
from src.models.database import Base, engine
from src.models import SessionLocal
from src.vectorstore.bm25_store import rebuild_bm25_index

from src.agents.graph import graph

app = FastAPI()

app.include_router(router)

Base.metadata.create_all(bind=engine)


@app.on_event("startup")
async def startup():

    db = SessionLocal()

    try:
        rebuild_bm25_index(db)

    finally:
        db.close()

    