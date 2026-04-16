<div align="center">

```
███╗   ███╗███████╗██████╗ ██████╗  ██████╗ ████████╗
████╗ ████║██╔════╝██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝
██╔████╔██║█████╗  ██║  ██║██████╔╝██║   ██║   ██║   
██║╚██╔╝██║██╔══╝  ██║  ██║██╔══██╗██║   ██║   ██║   
██║ ╚═╝ ██║███████╗██████╔╝██████╔╝╚██████╔╝   ██║   
╚═╝     ╚═╝╚══════╝╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   
```

### 🩺 AI-Powered Medical Knowledge Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Free_API-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-FF6B35?style=for-the-badge)](https://chainlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**Ask medical questions in any language. Get instant, grounded answers — powered by LLaMA 3 via Groq.**

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [✨ Features](#-features) · [🤝 Contributing](#-contributing)

---

</div>

## 🌟 What is MedBot?

MedBot is a **production-grade, enterprise-level medical RAG chatbot** that combines the power of:

- **Groq's blazing-fast inference** (LLaMA 3, Mixtral, Gemma2) — responses in under 1 second
- **FAISS semantic vector search** — finds the most relevant information from your medical knowledge base
- **Multi-source ingestion** — loads knowledge from both PDFs and live websites via Scrapy
- **10-language support** — ask in Hindi, Arabic, Urdu, French and more; get answers back in the same language
- **Enterprise safety layer** — detects crisis and emergency queries and routes them appropriately

> ⚠️ **Medical Disclaimer:** This tool is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## ✨ Features

### 🤖 Multi-Model Support (via Groq — Free & Fast)

| Model | Speed | Best For |
|---|---|---|
| `llama3-8b-8192` | ⚡ Fastest | General medical Q&A |
| `llama3-70b-8192` | 🧠 Smartest | Complex clinical reasoning |
| `mixtral-8x7b-32768` | 📚 Long context | Summarizing large documents |
| `gemma2-9b-it` | 🎯 Balanced | Friendly, educational answers |

Switch models **live from the UI** — no restart needed.

---

### 🌐 10-Language Real-Time Translation

Ask in your native language, get answers back in the same language:

| Language | Code | Language | Code |
|---|---|---|---|
| English | `en` | Arabic | `ar` |
| Hindi | `hi` | Portuguese | `pt` |
| French | `fr` | Urdu | `ur` |
| Spanish | `es` | Bengali | `bn` |
| German | `de` | Chinese | `zh-CN` |

Auto-detects your input language — no manual selection needed.

---

### 🎛️ Live Settings Panel (UI)

All settings are configurable from the ⚙️ panel — no code changes, no restarts:

| Setting | Options |
|---|---|
| 🌐 Response Language | 10 languages |
| 🩺 Response Tone | Clinical · Friendly · Educational |
| 🤖 Model | LLaMA3-8B · LLaMA3-70B · Mixtral · Gemma2 |
| 🌡️ Temperature | 0.0 → 1.0 slider |
| 📚 Top-K Documents | 2 → 10 slider |
| 📊 Confidence Score | Toggle on/off |
| 🔍 Auto-detect Language | Toggle on/off |

---

### 🛡️ Enterprise Safety Guard

Two-layer automatic safety detection:

**Layer 1 — Crisis Detection** (self-harm, overdose keywords)
→ Returns crisis helpline numbers (iCall India, findahelpline.com) and stops processing

**Layer 2 — Emergency Detection** (chest pain, heart attack, seizure keywords)
→ Returns emergency numbers (112/911/999) and urges immediate action

---

### 📊 Confidence Scoring

Every answer includes a confidence badge based on cosine similarity between your query and retrieved documents:

```
🟢 High confidence (87%)     — reliable answer from knowledge base
🟡 Medium confidence (61%)   — answer found, verify with a doctor
🔴 Low confidence (34%)      — knowledge base may not cover this topic
```

---

### 📋 Session Logging

Every session is automatically logged to `logs/session_<id>.log`:

```json
{
  "query": "What are symptoms of diabetes?",
  "detected_lang": "en",
  "tone": "Simple & Friendly",
  "model": "llama3-8b-8192",
  "temperature": 0.5,
  "confidence": 0.8241,
  "answer_preview": "Common symptoms of Type 2 diabetes include..."
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE INGESTION                      │
│                                                              │
│  📄 PDF Files          🌐 Websites (Scrapy)                  │
│  └─ PyPDFLoader        └─ Custom Spider                      │
│       │                      │                              │
│       └──────────┬───────────┘                              │
│                  ▼                                           │
│         RecursiveCharacterTextSplitter                       │
│         (chunk_size=500, overlap=50)                         │
│                  │                                           │
│                  ▼                                           │
│    HuggingFace Embeddings (all-MiniLM-L6-v2)                │
│                  │                                           │
│                  ▼                                           │
│         FAISS Vector Store  ──────► Saved to disk           │
└──────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                              │
│  User Query                                                  │
│      │                                                       │
│      ▼                                                       │
│  1. Safety Check ──► Crisis/Emergency → Return helpline      │
│      │                                                       │
│      ▼                                                       │
│  2. Language Detection + Translate to English                │
│      │                                                       │
│      ▼                                                       │
│  3. FAISS Retrieval (Top-K docs)                            │
│      │                                                       │
│      ▼                                                       │
│  4. Cosine Reranking (Top-2 most relevant)                  │
│      │                                                       │
│      ▼                                                       │
│  5. Confidence Score Calculation                            │
│      │                                                       │
│      ▼                                                       │
│  6. Groq LLM (LLaMA3 / Mixtral / Gemma2)                   │
│      │                                                       │
│      ▼                                                       │
│  7. Translate Answer → Target Language                       │
│      │                                                       │
│      ▼                                                       │
│  8. Log to session file                                      │
│      │                                                       │
│      ▼                                                       │
│  Chainlit UI Response                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
MedBot/
│
├── app.py                        # 🚀 Main Chainlit application
│
├── ingest.py                     # 📥 Data ingestion pipeline
│   ├── PDF loader (PyPDFLoader)
│   ├── Scrapy web scraper
│   ├── Text chunker
│   └── FAISS index builder
│
├── scraper/                      # 🕷️ Scrapy spider for web ingestion
│   ├── spider.py                 # Custom medical website spider
│   └── settings.py
│
├── vectorstores/
│   └── db_faiss/                 # 🗄️ FAISS vector index (auto-generated)
│
├── data/                         # 📚 Your PDF medical documents go here
│
├── logs/                         # 📋 Auto-generated session logs
│
├── .env                          # 🔑 API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medbot.git
cd medbot
```

### 2. Create and activate virtual environment

```bash
python3 -m venv env
source env/bin/activate        # Linux / Mac
env\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

Get your **free** Groq API key at 👉 [console.groq.com](https://console.groq.com) — no card required.

### 5. Add your medical data

**Option A — PDF files:**
Drop your PDF files into the `data/` folder.

**Option B — Websites (Scrapy):**
Edit `scraper/spider.py` and add your target medical websites.

### 6. Build the knowledge base

```bash
python ingest.py
```

This scrapes websites, loads PDFs, chunks the text, generates embeddings, and saves the FAISS index to `vectorstores/db_faiss/`.

### 7. Run the app

```bash
chainlit run app.py -w
```

Open your browser at **`http://localhost:8000`** 🎉

---

## 📥 Data Ingestion Pipeline

MedBot supports two ingestion sources that are merged into a single FAISS index:

### 📄 PDF Ingestion

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
```

Place any medical PDFs (textbooks, guidelines, research papers) in the `data/` folder and run `ingest.py`.

### 🕷️ Web Scraping via Scrapy

MedBot includes a Scrapy spider that cleanly extracts text from medical websites — stripping ads, navigation, and boilerplate:

```python
# scraper/spider.py
class MedicalSpider(scrapy.Spider):
    name = "medical"
    start_urls = [
        "https://www.mayoclinic.org/diseases-conditions",
        "https://medlineplus.gov/",
        # Add your sources here
    ]

    def parse(self, response):
        # Extracts clean paragraph text only
        text = " ".join(response.css("p::text").getall())
        yield {"url": response.url, "content": text}
```

Run the spider:
```bash
cd scraper
scrapy crawl medical -o output.json
```

Then `ingest.py` picks up `output.json` automatically and merges it with your PDFs.

### 🗄️ FAISS Index Building

```
All documents
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=500, overlap=50)
    │
    ▼
HuggingFace Embeddings
(sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
FAISS.from_documents()
    │
    ▼
Saved → vectorstores/db_faiss/
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key from console.groq.com |
| `GROQ_MODEL` | `llama3-8b-8192` | Default model (overridable from UI) |

---

## 📦 Requirements

```txt
langchain
langchain-community
langchain-groq
langchain-core
faiss-cpu
sentence-transformers
chainlit
deep-translator
langdetect
python-dotenv
scrapy
pypdf
scikit-learn
numpy
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

### Reporting Issues

Open a [GitHub Issue](../../issues) with:
- Your OS and Python version
- The exact error message
- Steps to reproduce

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit clearly:
   ```bash
   git commit -m "feat: add XYZ feature"
   ```
4. Push and open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python style
- Never commit `.env`, model files, or `vectorstores/`
- Add docstrings to new functions
- Update this README if you add features

---

## 🗺️ Roadmap

- [ ] Voice input support
- [ ] Image/X-ray analysis via multimodal models
- [ ] User authentication for multi-user deployment
- [ ] Docker containerization
- [ ] REST API wrapper
- [ ] Feedback loop to improve retrieval quality

---

## 🙏 Acknowledgements

| Library | Purpose |
|---|---|
| [Groq](https://groq.com) | Ultra-fast LLM inference |
| [LangChain](https://langchain.com) | RAG pipeline framework |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [Chainlit](https://chainlit.io) | Chat UI framework |
| [Scrapy](https://scrapy.org) | Web scraping |
| [HuggingFace](https://huggingface.co) | Sentence embeddings |
| [deep-translator](https://github.com/nidhaloff/deep-translator) | Multi-language translation |

---

<div align="center">

**Built with ❤️ for accessible medical AI**

*If this project helped you, please ⭐ star the repository!*

</div>
