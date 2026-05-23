from fastapi import FastAPI
from src.api.routes import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def home():
    print("RAG app is running")
    return {
        "message": "RAG app running"
    }

