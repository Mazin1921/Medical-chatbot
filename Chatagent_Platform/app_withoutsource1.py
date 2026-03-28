import chainlit as cl

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from langdetect import detect

import numpy as np
import datetime
import logging
import json
import os
import re

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DB_FAISS_PATH = 'vectorstores/db_faiss'
MODEL_PATH = 'C:\\Users\\HP5CD\\OneDrive\\Desktop\\Mazin documents\\GEN-AI-PROJECT\\Llama2-Medical-Chatbot\\llama-2-7b-chat.ggmlv3.q2_K.bin'

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LANGUAGES
# ─────────────────────────────────────────────────────────────

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Arabic": "ar",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh-CN",
    "Portuguese": "pt",
    "Urdu": "ur",
    "Bengali": "bn",
}

# ─────────────────────────────────────────────────────────────
# SETTINGS UI
# ─────────────────────────────────────────────────────────────

settings_config = [
    cl.input_widget.Select(
        id="language",
        label="🌐 Response Language",
        values=list(LANGUAGES.keys()),
        initial_value="English",
    ),
    cl.input_widget.Select(
        id="tone",
        label="🩺 Response Tone",
        values=["Clinical & Precise", "Simple & Friendly", "Detailed & Educational"],
        initial_value="Simple & Friendly",
    ),
    cl.input_widget.Slider(
        id="temperature",
        label="🌡️ Creativity",
        initial=0.5,
        min=0.1,
        max=1.0,
        step=0.1,
    ),
    cl.input_widget.Slider(
        id="top_k",
        label="📚 Top-K Docs",
        initial=5,
        min=2,
        max=10,
        step=1,
    ),
]

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────

def get_logger(session_id):
    logger = logging.getLogger(session_id)
    if not logger.handlers:
        file = os.path.join(LOG_DIR, f"{session_id}.log")
        handler = logging.FileHandler(file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

PROMPT = """You are a helpful medical assistant.

Context:
{context}

Question:
{question}

Answer:"""

def get_prompt():
    return PromptTemplate(template=PROMPT, input_variables=["context", "question"])

# ─────────────────────────────────────────────────────────────
# SAFETY
# ─────────────────────────────────────────────────────────────

def safety_check(q):
    if re.search(r"(suicide|kill myself|overdose)", q, re.I):
        return "⚠️ Please contact a helpline immediately. India: 9152987821"
    return None

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_en(text, src):
    if src == "en":
        return text
    try:
        return GoogleTranslator(source=src, target="en").translate(text)
    except:
        return text

def translate_from_en(text, tgt):
    if tgt == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=tgt).translate(text)
    except:
        return text

# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_db(emb):
    return FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)

def load_llm(temp):
    return CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        temperature=temp,
        max_new_tokens=512
    )

# ─────────────────────────────────────────────────────────────
# CHAT START
# ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    embeddings = load_embeddings()
    db = load_db(embeddings)

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(session_id)

    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("db", db)
    cl.user_session.set("logger", logger)
    cl.user_session.set("settings", {})

    await cl.ChatSettings(settings_config).send()

    await cl.Message(content="👋 Welcome to MedBot!").send()

# ─────────────────────────────────────────────────────────────
# SETTINGS UPDATE
# ─────────────────────────────────────────────────────────────

@cl.on_settings_update
async def update(settings):
    cl.user_session.set("settings", settings)

# ─────────────────────────────────────────────────────────────
# MAIN MESSAGE
# ─────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):

    embeddings = cl.user_session.get("embeddings")
    db = cl.user_session.get("db")
    logger = cl.user_session.get("logger")
    settings = cl.user_session.get("settings") or {}

    query = message.content.strip()
    if not query:
        return

    # safety
    safety = safety_check(query)
    if safety:
        await cl.Message(content=safety).send()
        return

    # language
    detected = detect_lang(query)
    query_en = translate_to_en(query, detected)

    # retrieve
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = await retriever.ainvoke(query_en)

    context = "\n".join([d.page_content for d in docs])

    # LLM
    llm = load_llm(settings.get("temperature", 0.5))
    chain = LLMChain(llm=llm, prompt=get_prompt())

    answer_en = await chain.arun({
        "context": context,
        "question": query_en
    })

    # translate back
    target_lang = LANGUAGES.get(settings.get("language", "English"), "en")
    answer = translate_from_en(answer_en, target_lang)

    await cl.Message(content=answer).send()