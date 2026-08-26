"""
Módulo 2-3: Cadena RAG (retriever + LLM).

Objetivo: dado un vector store ya construido (ingest.py), armar la cadena
que recupera los chunks relevantes y genera una respuesta citando el
proyecto/código de origen.

Corresponde a los módulos "RAG using LangChain" y "LCEL Chains" del curso.

TODO (a medida que avance en el curso):
1. Cargar el vector store persistido (Chroma) como retriever.
2. Definir el prompt: debe instruir al modelo a responder SOLO con base en
   los documentos recuperados y citar de qué proyecto viene la información.
3. Armar la chain con LCEL (retriever | prompt | llm | output_parser).
4. (Opcional, módulo LangGraph) añadir memoria conversacional.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def load_retriever():
    """TODO: cargar el vector store persistido y devolverlo como retriever."""
    raise NotImplementedError


def build_rag_chain():
    """TODO: construir la chain completa (retriever | prompt | llm)."""
    raise NotImplementedError


def ask(question: str) -> str:
    """Punto de entrada: recibe una pregunta, devuelve la respuesta del RAG."""
    chain = build_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    pregunta = "¿Qué modelo usé para la detección de fraude en SENTINEL NovaPay?"
    print(ask(pregunta))
