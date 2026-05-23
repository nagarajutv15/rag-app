from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="RAG App"
)

app.include_router(router)


@app.get("/")
async def home():
    return {
        "message": "RAG app running"
    }