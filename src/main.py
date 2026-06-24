from fastapi import FastAPI
from src.api.routes import router
from src.models.database import Base, engine
from src.models import SessionLocal
from src.rag.vectorstore.bm25_store import rebuild_bm25_index
from src.models.organization_seed import seed_organization_data
from src.api.organization import router as organization_router
from src.agents.graph_builder import graph

app = FastAPI()

app.include_router(router)
app.include_router(organization_router)

Base.metadata.create_all(bind=engine)


@app.on_event("startup")
async def startup():

    db = SessionLocal()

    try:
        rebuild_bm25_index(db)
        seed_organization_data(db)

    finally:
        db.close()

    result = graph.invoke(
        {
            "question":
            "Hey who is Nagaraju and may i know who is he manager and in which project he is working and which client and what assest Id taht he own?"
        }
    )

    print(result)