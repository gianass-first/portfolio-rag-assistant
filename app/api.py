"""
Etapa final: exponer la cadena RAG como API con FastAPI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import ask

app = FastAPI(title="Portfolio RAG Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(payload: Question):
    answer = ask(payload.question)
    return {"answer": answer}
