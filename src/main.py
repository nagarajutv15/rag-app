from fastapi import FastAPI
from src.api.routes import router
from src.models.database import Base, engine
from src.models.database import SessionLocal
from src.rag.vectorstore.bm25_store import rebuild_bm25_index


app = FastAPI()

app.include_router(router)

Base.metadata.create_all(bind=engine)


# Startup Event: Rebuild BM25 Index on Application Startup
@app.on_event("startup")
def startup():

    db = SessionLocal()

    try:

        rebuild_bm25_index(db)

    finally:

        db.close()
