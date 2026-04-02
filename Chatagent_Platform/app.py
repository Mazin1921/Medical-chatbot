from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import detect
import chainlit as cl
import numpy as np
import datetime
import asyncio
import logging
import json
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# LOAD .env
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. "
        "Please create a .env file with: GOOGLE_API_KEY=your_key_here"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_FAISS_PATH = 'vectorstores/db_faiss'
LOG_DIR       = 'logs'
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # ← default is lite (highest free quota)
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGES = {
    "English":    "en",
    "Hindi":      "hi",
    "French":     "fr",
    "Arabic":     "ar",
    "Spanish":    "es",
    "German":     "de",
    "Chinese":    "zh-CN",
    "Portuguese": "pt",
    "Urdu":       "ur",
    "Bengali":    "bn",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHAINLIT SETTINGS PANEL
# ─────────────────────────────────────────────────────────────────────────────

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
    cl.input_widget.Select(
        id="gemini_model",
        label="🤖 Gemini Model",
        values=["gemini-1.5-flash",
        "gemini-2.0-flash"],
        initial_value="gemini-1.5-flash",   # ← FIXED: lite first and default
    ),
    cl.input_widget.Slider(
        id="temperature",
        label="🌡️ Creativity (Temperature)",
        initial=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    cl.input_widget.Slider(
        id="top_k",
        label="📚 Source Docs to Retrieve (Top-K)",
        initial=5,
        min=2,
        max=10,
        step=1,
    ),
    cl.input_widget.Switch(
        id="show_confidence",
        label="📊 Show Confidence Score",
        initial=True,
    ),
    cl.input_widget.Switch(
        id="auto_detect_lang",
        label="🔍 Auto-Detect Input Language",
        initial=True,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_session_logger(session_id: str) -> logging.Logger:
    logger = logging.getLogger(session_id)
    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, f"session_{session_id}.log")
        handler  = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "Clinical & Precise": """You are a clinical medical assistant. Use the context to give a precise, professional answer.
If you don't know the answer, say so — never fabricate.

Context: {context}
Question: {question}

Clinical Answer:""",

    "Simple & Friendly": """You are a friendly, empathetic medical assistant helping a non-expert user.
Use the context below to answer in simple, warm language. If unsure, say so honestly.

Context: {context}
Question: {question}

Helpful Answer:""",

    "Detailed & Educational": """You are a detailed medical educator. Use the context to give a thorough explanation
including causes, mechanisms, and takeaways. Never fabricate information.

Context: {context}
Question: {question}

Educational Answer:""",
}

def get_prompt(tone: str) -> PromptTemplate:
    template = PROMPT_TEMPLATES.get(tone, PROMPT_TEMPLATES["Simple & Friendly"])
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY GUARD
# ─────────────────────────────────────────────────────────────────────────────

DANGER_PATTERNS = re.compile(
    r"\b(suicide|overdose|self.harm|kill myself|end my life|lethal dose)\b",
    re.IGNORECASE,
)
EMERGENCY_KEYWORDS = re.compile(
    r"\b(chest pain|can't breathe|cannot breathe|heart attack|stroke|unconscious|seizure|severe bleeding)\b",
    re.IGNORECASE,
)

def safety_check(query: str) -> str | None:
    if DANGER_PATTERNS.search(query):
        return (
            "⚠️ **Crisis Support** — If you or someone you know is in distress, "
            "please contact a crisis helpline immediately:\n"
            "- **India:** iCall — 9152987821\n"
            "- **Global:** findahelpline.com\n\n"
            "I'm not able to assist with this topic, but you are not alone. 💙"
        )
    if EMERGENCY_KEYWORDS.search(query):
        return (
            "🚨 **This sounds like a medical emergency.** "
            "Please call **112 (India) / 911 (US) / 999 (UK)** or go to your nearest emergency room immediately.\n\n"
            "I cannot replace emergency medical care."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception:
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(query: str, docs: list, embeddings) -> float:
    if not docs:
        return 0.0
    q_emb  = embeddings.embed_query(query)
    d_embs = embeddings.embed_documents([d.page_content for d in docs])
    sims   = cosine_similarity([q_emb], d_embs)[0]
    return float(np.mean(sims))

def confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 High confidence ({score:.0%})"
    elif score >= 0.50:
        return f"🟡 Medium confidence ({score:.0%})"
    else:
        return f"🔴 Low confidence ({score:.0%}) — please verify with a doctor"

# ─────────────────────────────────────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────────────────────────────────────

def rerank_documents(query: str, documents: list, embeddings, top_k: int = 2) -> list:
    q_emb  = embeddings.embed_query(query)
    d_embs = embeddings.embed_documents([d.page_content for d in documents])
    sims   = cosine_similarity([q_emb], d_embs)[0]
    ranked = sorted(zip(documents, sims), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LLM LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_llm(temperature: float = 0.5, model_name: str = GEMINI_MODEL):
    clean_model = model_name.replace("models/", "")
    return ChatGoogleGenerativeAI(
        model=clean_model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

def load_db(embeddings):
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE QUERY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(query: str, settings: dict, embeddings, db, logger) -> dict:
    # 1. Safety
    safety_msg = safety_check(query)
    if safety_msg:
        return {"answer": None, "confidence": 0.0, "safety_msg": safety_msg, "detected_lang": "en"}

    # 2. Language
    auto_detect   = settings.get("auto_detect_lang", True)
    chosen_lang   = LANGUAGES.get(settings.get("language", "English"), "en")
    detected_lang = detect_language(query) if auto_detect else chosen_lang
    query_en      = translate_to_english(query, detected_lang)

    # 3. Retrieve & rerank
    top_k       = int(settings.get("top_k", 5))
    retriever   = db.as_retriever(search_kwargs={"k": top_k})
    raw_docs    = await retriever.ainvoke(query_en)
    ranked_docs = rerank_documents(query_en, raw_docs, embeddings, top_k=min(2, len(raw_docs)))

    # 4. Confidence
    confidence = compute_confidence(query_en, ranked_docs, embeddings)

    # 5. LLM — always use gemini-2.0-flash-lite from settings (never falls back to flash)
    temperature = float(settings.get("temperature", 0.5))
    tone        = settings.get("tone", "Simple & Friendly")
    model_name  = settings.get("gemini_model") or GEMINI_MODEL  # ← reads from UI, falls back to lite
    llm         = load_llm(temperature, model_name)
    prompt      = get_prompt(tone)
    context     = "\n\n".join([d.page_content for d in ranked_docs])
    chain       = prompt | llm | StrOutputParser()

    # Retry up to 3 times on quota exhaustion (429)
    answer_en = ""
    for attempt in range(3):
        try:
            answer_en = await chain.ainvoke({"context": context, "question": query_en})
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                await cl.Message(
                    content=f"⏳ Gemini quota limit hit. Retrying in {wait}s (attempt {attempt+1}/3)..."
                ).send()
                await asyncio.sleep(wait)
                if attempt == 2:
                    answer_en = "I'm currently rate-limited. Please wait a minute and try again."
            else:
                raise

    # 6. Translate back
    answer = translate_from_english(answer_en.strip(), chosen_lang)

    # 7. Log
    logger.info(json.dumps({
        "query":          query,
        "query_en":       query_en,
        "detected_lang":  detected_lang,
        "target_lang":    chosen_lang,
        "tone":           tone,
        "model":          model_name,
        "temperature":    temperature,
        "top_k":          top_k,
        "confidence":     round(confidence, 4),
        "answer_preview": answer[:120],
    }))

    return {
        "answer":        answer,
        "confidence":    confidence,
        "safety_msg":    None,
        "detected_lang": detected_lang,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHAINLIT LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)

@cl.on_chat_start
async def start():
    embeddings = load_embeddings()
    db         = load_db(embeddings)

    session_id = cl.user_session.get("id") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger     = get_session_logger(session_id)
    logger.info("Session started")

    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("db",         db)
    cl.user_session.set("logger",     logger)
    cl.user_session.set("settings",   {})

    await cl.ChatSettings(settings_config).send()

    await cl.Message(
        author="🩺 MedBot",
        content=(
            "---\n"
            "## ⚠️ Disclaimer\n"
            "> This assistant provides **educational information only** and is **not a substitute** "
            "for professional medical advice, diagnosis, or treatment.\n"
            "> Always consult a qualified healthcare provider for personal health concerns.\n"
            "---"
        ),
    ).send()

    await cl.Message(
        content=(
            "👋 Welcome to **MedBot** — powered by ✨ Google Gemini!\n\n"
            "🌐 Use the **settings panel** (⚙️ top-right) to choose your **language**, "
            "**tone**, **Gemini model**, and other preferences.\n\n"
            "Ask me anything about symptoms, conditions, medications, or general health!"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    embeddings = cl.user_session.get("embeddings")
    db         = cl.user_session.get("db")
    logger     = cl.user_session.get("logger")
    settings   = cl.user_session.get("settings") or {}

    if message.elements:
        await cl.Message(content="📸 Image analysis is not yet supported. Please describe your query in text.").send()
        return

    query = message.content.strip()
    if not query:
        return

    async with cl.Step(name="✨ Asking Gemini...") as step:
        result = await run_pipeline(query, settings, embeddings, db, logger)
        step.output = "Done"

    if result["safety_msg"]:
        await cl.Message(content=result["safety_msg"]).send()
        return

    answer     = result["answer"]
    confidence = result["confidence"]
    show_conf  = settings.get("show_confidence", True)
    auto_det   = settings.get("auto_detect_lang", True)
    det_lang   = result["detected_lang"]
    chosen_key = settings.get("language", "English")
    chosen_code= LANGUAGES.get(chosen_key, "en")

    lines = [answer]

    if show_conf:
        lines.append(f"\n\n---\n{confidence_badge(confidence)}")

    if auto_det and det_lang != chosen_code:
        lines.append(
            f"\n🌐 *Detected input language:* `{det_lang}` → *responding in* `{chosen_key}`"
        )

    lines.append("\n\n> ⚕️ *Always verify medical information with a licensed healthcare professional.*")

    await cl.Message(content="".join(lines)).send()