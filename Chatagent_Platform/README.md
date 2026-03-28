# 🏥 Llama2 Medical Chatbot

A locally-running AI-powered medical chatbot built with **Meta's LLaMA-2**, **FAISS** vector search, and a sleek **Chainlit** UI — giving you a private, offline-capable assistant for medical knowledge queries.

> ⚠️ **Disclaimer:** This chatbot is for **informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **LLaMA-2 (7B)** | Runs locally via GGML quantized model — no API keys, no data sent to the cloud |
| 🔍 **FAISS Vector Search** | Fast semantic retrieval over medical documents using Meta's FAISS library |
| 💬 **Chainlit UI** | Clean, chat-style interface with real-time streaming responses |
| 🔒 **Privacy First** | Fully local inference — your medical queries never leave your machine |
| 📚 **Document-Grounded** | Answers are grounded in your own medical knowledge base, not just model weights |

---

## 🎯 Use Cases

- Querying symptoms and general medical terminology
- Searching and summarizing medical documents or PDFs
- Building a private health knowledge assistant for clinics or personal use
- Experimenting with local LLM + RAG (Retrieval-Augmented Generation) pipelines

---

## 🛠️ Tech Stack

- **LLM:** [LLaMA-2 7B Chat (GGML)](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) by Meta
- **UI Framework:** [Chainlit](https://github.com/Chainlit/chainlit)
- **Language:** Python 3.x

---

## 📸 Demo

> _Screenshots or a screen recording of the chatbot in action go here._

```
[ Add a GIF or screenshot of your chatbot interface here ]
```

Example interaction:

```
User:   What are the common symptoms of Type 2 diabetes?

Bot:    Common symptoms of Type 2 diabetes include increased thirst,
        frequent urination, unexplained weight loss, fatigue, blurred
        vision, and slow-healing sores. Some people may have no symptoms
        at all in the early stages...
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ubuntu / macOS (Windows via WSL)
- ~8 GB RAM minimum (16 GB recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llama2-medical-chatbot.git
cd llama2-medical-chatbot
```

### 2. Create & Activate a Virtual Environment

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the LLaMA-2 Model

Download the quantized GGML model from Hugging Face:

👉 [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

Place the downloaded `.bin` file in the `models/` directory:

```
llama2-medical-chatbot/
├── models/
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin
├── app.py
├── requirements.txt
└── ...
```

### 5. Run the App

```bash
chainlit run app.py -w
```

Then open your browser at **`http://localhost:8000`** 🎉

---

## 📁 Project Structure

```
llama2-medical-chatbot/
│
├── app.py                  # Main Chainlit app entry point
├── requirements.txt        # Python dependencies
├── models/                 # Place your downloaded LLaMA-2 model here
├── data/                   # Your medical documents / PDFs for the knowledge base
├── vectorstore/            # FAISS index (auto-generated after first run)
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get involved:

### Reporting Issues

- Use the [GitHub Issues](../../issues) tab to report bugs or suggest features.
- Please include your OS, Python version, and steps to reproduce the issue.

### Submitting Pull Requests

1. **Fork** the repository
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and write clear commit messages
4. **Test** your changes locally before submitting
5. **Open a Pull Request** with a description of what you changed and why

### Development Guidelines

- Keep code readable and well-commented
- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python style
- Don't commit model files or large binaries — add them to `.gitignore`
- Update the README if you add new features or change setup steps

---

## 📄 License

This project is open-source. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Meta AI](https://ai.meta.com/) for LLaMA-2 and FAISS
- [TheBloke](https://huggingface.co/TheBloke) for the quantized GGML model weights
- [Chainlit](https://chainlit.io/) for the chat UI framework

---

<p align="center">Made with ❤️ for accessible medical AI</p>
