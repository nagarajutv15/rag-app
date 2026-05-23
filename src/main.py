from fastapi import FastAPI
from src.api.routes import router
from src.models.database import engine
from src.models.init_db import initialize_database
from src.models.seed import seed_data

app = FastAPI()
app.include_router(router)

@app.get("/")
async def home():
    print("RAG app is running")
    return {
        "message": "RAG app running"
    }

@app.on_event("startup")
async def startup():

    try:
        initialize_database()

        print(" Database Connected Successfully")


    except Exception as e:
        print(f" Database Connection Failed : {e}")