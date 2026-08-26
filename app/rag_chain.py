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
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

RAG_PROMPT = """Eres un asistente que responde preguntas sobre el portafolio de proyectos de Data Science de Gian Marco.

Responde SOLO con base en el siguiente contexto extraído de sus proyectos. Si el contexto no contiene información suficiente para responder, dilo claramente, no inventes datos.

Cuando respondas, menciona de qué proyecto proviene la información.

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


def build_rag_chain():
    """Construye la chain completa (retriever | prompt | llm | output_parser)."""
    retriever = load_retriever()
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> str:
    """Punto de entrada: recibe una pregunta, devuelve la respuesta del RAG."""
    chain = build_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    pregunta = "¿Qué modelo usé para la detección de fraude en SENTINEL NovaPay?"
    print(ask(pregunta))
