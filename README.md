# Portfolio RAG Assistant

Asistente conversacional tipo RAG (Retrieval-Augmented Generation) que responde preguntas sobre mis propios proyectos de Data Science / ML (SENTINEL NovaPay, Absenteeism at Work, World Marathon Majors EDA, Boston Marathon Qualification Model), citando el código y los resultados reales de cada uno.

Proyecto de aprendizaje práctico en paralelo al curso de **LangChain Academy**, construido de forma incremental a medida que avanzo en el curso.

## Objetivo

- Aplicar conceptos de RAG (embeddings, vector store, retrieval, chains) sobre datos propios y reales.
- Servir como pieza de portafolio que demuestra IA generativa aplicada, además de exponer mis otros proyectos.

## Stack

- **LangChain** — orquestación del pipeline RAG
- **ChromaDB** — vector store local (gratuito, sin infraestructura externa)
- **FastAPI** — API de consulta
- **Streamlit** — interfaz de chat (mismo patrón que usé en Absenteeism at Work)

## Estructura

```
portfolio-rag-assistant/
├── app/
│   ├── ingest.py        # Carga documentos, genera embeddings, los guarda en Chroma
│   ├── rag_chain.py      # Define la cadena RAG (retriever + LLM)
│   ├── api.py             # Endpoints FastAPI
│   └── streamlit_app.py   # Interfaz de chat
├── data/
│   └── portfolio_docs/    # READMEs/notas de mis proyectos (fuente de conocimiento del RAG)
├── requirements.txt
└── .env.example
```

## Roadmap de construcción (en paralelo al curso)

- [ ] **Módulo 1 (LangChain Academy — fundamentos):** `ingest.py` — carga de documentos y splitting
- [ ] **Módulo 2 (embeddings + vector store):** generar embeddings, indexar en Chroma
- [ ] **Módulo 3 (retrieval + chains):** `rag_chain.py` — retriever + prompt + LLM
- [ ] **Módulo 4 (LangGraph, si aplica):** lógica conversacional con memoria
- [ ] **Integración final:** exponer todo vía `api.py` (FastAPI) y `streamlit_app.py`

## Cómo correrlo (una vez completado)

```bash
pip install -r requirements.txt
cp .env.example .env  # agregar tu API key
python app/ingest.py
uvicorn app.api:app --reload
# o bien:
streamlit run app/streamlit_app.py
```
