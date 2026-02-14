from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_core import ask_rag
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from app.rag_core import ask_rag_stream





app = FastAPI(title="Sentinel RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    result = ask_rag(req.question)   # result is a dict now
    return {"question": req.question, **result}


@app.post("/ask-stream")
def ask_stream(req: AskRequest):
    def event_generator():
        for token in ask_rag_stream(req.question):
            yield {"data": token}

    return EventSourceResponse(event_generator())