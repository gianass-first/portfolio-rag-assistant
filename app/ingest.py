"""
Módulo 1: Ingesta de documentos.

Objetivo: cargar los README/notas de mis proyectos (data/portfolio_docs/),
dividirlos en chunks, generar embeddings y guardarlos en un vector store (Chroma).

Corresponde al módulo de "Core Components de LangChain" del curso de LangChain Academy.

TODO (a medida que avance en el curso):
1. Cargar documentos desde data/portfolio_docs/ con un DocumentLoader de LangChain.
2. Dividir el texto en chunks con un TextSplitter (ej. RecursiveCharacterTextSplitter).
3. Generar embeddings (OpenAIEmbeddings o alternativa).
4. Persistir los chunks + embeddings en Chroma (CHROMA_PERSIST_DIR en .env).
"""

import os
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "data/portfolio_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def load_documents():
    """TODO: cargar todos los archivos .md/.txt de DOCS_PATH."""
    raise NotImplementedError


def split_documents(documents):
    """TODO: dividir documentos en chunks manejables."""
    raise NotImplementedError


def build_vector_store(chunks):
    """TODO: generar embeddings e indexar en Chroma, persistiendo en PERSIST_DIR."""
    raise NotImplementedError


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print(f"Ingesta completa. Vector store persistido en {PERSIST_DIR}")
