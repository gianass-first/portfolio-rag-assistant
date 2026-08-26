"""
Módulo 2-3: Cadena RAG (retriever + LLM).

Objetivo: dado un vector store ya construido (ingest.py), armar la cadena
que recupera los chunks relevantes y genera una respuesta citando el
proyecto/código de origen.

Corresponde a los módulos "RAG using LangChain" y "LCEL Chains" del curso.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

RAG_PROMPT = """Eres un asistente que responde preguntas sobre el portafolio de proyectos de Data Science de Gian Marco.

Responde SOLO con base en el siguiente contexto extraído de sus proyectos. Si el contexto no contiene información suficiente para responder, dilo claramente, no inventes datos.

Cuando respondas, menciona de qué proyecto proviene la información.

Ten en cuenta el historial de la conversación para entender preguntas de seguimiento (por ejemplo, si el usuario dice "y ese otro proyecto" o "cuéntame más sobre eso").

Historial de la conversación:
{chat_history}

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def load_retriever():
    """Carga el vector store persistido y lo devuelve como retriever."""
    embeddings = VoyageAIEmbeddings(model="voyage-3.5")
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Convierte los documentos recuperados en un solo bloque de texto para el prompt."""
    return "\n\n---\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}"
        for doc in docs
    )


def format_history(messages):
    """Convierte el historial de mensajes (lista de dicts role/content) en texto plano."""
    if not messages:
        return "(sin mensajes previos)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_rag_chain():
    """Construye la chain completa (retriever | prompt | llm | output_parser)."""
    retriever = load_retriever()
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_history(x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, chat_history: list = None) -> str:
    """Punto de entrada: recibe una pregunta y opcionalmente el historial previo."""
    chain = build_rag_chain()
    return chain.invoke({"question": question, "chat_history": chat_history or []})


if __name__ == "__main__":
    pregunta = "¿Qué modelo usé para la detección de fraude en SENTINEL NovaPay?"
    print(ask(pregunta))