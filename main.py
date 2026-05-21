def main():
    print("Hello from ragapp!")
    print("This is a sample application to demonstrate how to use ragapp.")


if __name__ == "__main__":
    main()

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG app is running"}