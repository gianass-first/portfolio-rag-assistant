"""
Etapa final: interfaz de chat en Streamlit, mismo patrón usado en
Absenteeism at Work.
"""

import streamlit as st
from app.rag_chain import ask

st.set_page_config(page_title="Portfolio RAG Assistant")
st.title("Pregúntale a mi portafolio")

pregunta = st.text_input("Escribe tu pregunta sobre mis proyectos de Data Science:")

if pregunta:
    with st.spinner("Buscando en el portafolio..."):
        respuesta = ask(pregunta)
    st.write(respuesta)
