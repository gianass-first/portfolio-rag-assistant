"""
Etapa final: interfaz de chat en Streamlit, mismo patrón usado en
Absenteeism at Work.
"""

import streamlit as st
from rag_chain import build_rag_chain

st.set_page_config(page_title="Portfolio RAG Assistant", page_icon="💼", layout="centered")

# Paleta inspirada en el banner de mi LinkedIn: acentos azul marino/cian sobre fondo blanco
NAVY_DARK = "#0A1929"
NAVY_TEXT = "#0D2B45"
ELECTRIC_CYAN = "#00A8CC"
ELECTRIC_BLUE = "#2E9BF0"

st.markdown(
    f"""
    <style>
    .main-header {{
        background: linear-gradient(90deg, {NAVY_DARK}, {ELECTRIC_BLUE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    [data-testid="stChatMessageAvatarUser"] {{
        background-color: {NAVY_TEXT} !important;
    }}
    [data-testid="stChatMessageAvatarAssistant"] {{
        background-color: {ELECTRIC_CYAN} !important;
    }}
    .stChatInput textarea {{
        border-color: {ELECTRIC_BLUE} !important;
    }}
    .stChatInput button {{
        background-color: {ELECTRIC_BLUE} !important;
        border-color: {ELECTRIC_BLUE} !important;
    }}
    hr {{
        border-color: {ELECTRIC_BLUE}55 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<h1 class="main-header">💼 Pregúntale a mi portafolio</h1>', unsafe_allow_html=True)
st.caption("Chatbot RAG sobre mis proyectos de Data Science — SENTINEL NovaPay · Absenteeism at Work · World Marathon Majors · Boston Marathon")
st.divider()


@st.cache_resource
def get_chain():
    """Construye la chain RAG una sola vez y la reutiliza entre interacciones."""
    return build_rag_chain()


rag_chain = get_chain()

# Historial de la conversación en esta sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial previo
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "💼"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Input tipo chat
pregunta = st.chat_input("Escribe tu pregunta sobre mis proyectos de Data Science...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(pregunta)

    with st.chat_message("assistant", avatar="💼"):
        with st.spinner("Buscando en el portafolio..."):
            respuesta = rag_chain.invoke({
                "question": pregunta,
                "chat_history": st.session_state.messages[:-1],
            })
        st.write(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})