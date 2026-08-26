"""
Módulo 1: Ingesta de documentos.

Objetivo: cargar los README/notas de mis proyectos (data/portfolio_docs/),
dividirlos en chunks, generar embeddings y guardarlos en un vector store (Chroma).

Corresponde al módulo de "Core Components de LangChain" del curso de LangChain Academy.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_PATH = "data/portfolio_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def load_documents():
    """Carga todos los archivos .md de DOCS_PATH como Documents de LangChain."""
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Cargados {len(documents)} documentos desde {DOCS_PATH}")
    return documents


def split_documents(documents):
    """Divide los documentos en chunks manejables para embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Generados {len(chunks)} chunks a partir de {len(documents)} documentos")
    return chunks


def build_vector_store(chunks):
    """Genera embeddings con Voyage AI e indexa en Chroma, persistiendo en PERSIST_DIR."""
    embeddings = VoyageAIEmbeddings(model="voyage-3.5")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vector_store


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print(f"Ingesta completa. Vector store persistido en {PERSIST_DIR}")