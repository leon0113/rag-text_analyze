from fastapi import FastAPI
from routes.query import router as query_router

app = FastAPI(
    title="My RAG Application",
    description="Simple RAG application using FastAPI and LangChain",
    version="0.1.0"
)

app.include_router(query_router)

@app.get("/")
def root():
    return {
        "message": "running"
    }
