"""
Etapa final: exponer la cadena RAG como API con FastAPI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import build_rag_chain

app = FastAPI(title="Portfolio RAG Assistant")

# La chain se construye una sola vez al arrancar el servidor,
# no en cada request (evita reconectar a Voyage/Chroma/Claude cada vez).
rag_chain = build_rag_chain()


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(payload: Question):
    answer = rag_chain.invoke(payload.question)
    return {"answer": answer}