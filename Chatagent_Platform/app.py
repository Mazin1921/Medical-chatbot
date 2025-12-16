from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.document import Document
from langchain.chains import RetrievalQA

from sklearn.metrics.pairwise import cosine_similarity
import chainlit as cl
import numpy as np
import os
from urllib.parse import urlparse  # to extract domains

DB_FAISS_PATH = 'vectorstores/db_faiss'

# 🔹 Prompt Template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return CTransformers(
        model='C:\\Users\\HP5CD\\OneDrive\\Desktop\\Mazin documents\\GEN-AI-PROJECT\\Llama2-Medical-Chatbot\\llama-2-7b-chat.ggmlv3.q2_K.bin',
        model_type="llama",
        max_new_tokens=256,
        temperature=0.5
    )

# 🔹 Custom reranker
def rerank_documents(query, documents, embedding_model, top_k=2):
    query_emb = embedding_model.embed_query(query)
    doc_texts = [doc.page_content for doc in documents]
    doc_embs = embedding_model.embed_documents(doc_texts)

    similarities = cosine_similarity([query_emb], doc_embs)[0]
    ranked_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs[:top_k]]

# 🔹 Main chain
def retrieval_qa_chain(llm, prompt, db, embeddings):
    retriever = db.as_retriever(search_kwargs={"k": 5})

    async def custom_chain(query):
        raw_docs = await retriever.ainvoke(query)
        ranked_docs = rerank_documents(query, raw_docs, embeddings, top_k=2)

        context = "\n\n".join([doc.page_content for doc in ranked_docs])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.arun({'context': context, 'question': query})

        # Extract unique domain names only
        unique_domains = set()
        for doc in ranked_docs:
            source_url = doc.metadata.get("source", "")
            if source_url:
                domain = urlparse(source_url).netloc
                if domain:
                    unique_domains.add(domain)

        return {
            "result": result,
            "sources": sorted(unique_domains)
        }

    return custom_chain

# 🔹 Load everything
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db, embeddings)

# 🔹 Chainlit start
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)

    disclaimer = cl.Message(
        author="🩺 MedicalBot",
        content="""
---
## ⚠️ **Disclaimer**
> 🧠 *This assistant is powered by AI and provides information for **educational** and **informational purposes only**.*
> 🚫 It is **not a substitute** for professional medical advice, diagnosis, or treatment.
> 🩺 Always consult a **qualified healthcare provider** for critical or personal health concerns.
---
"""
    )
    await disclaimer.send()

    welcome = cl.Message(content="👋 Hi, Welcome to **Medical Bot**! What is your query today?")
    await welcome.send()

# 🔹 Message handler
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True

    if message.elements:
        await cl.Message(content="📸 Currently image processing is under development.").send()
        return

    res = await chain(message.content)
    answer = res["result"]
    domains = res.get("sources", [])

    if domains:
        answer += "\n\nSources:\n" + "\n".join([f"- {domain}" for domain in domains])
    else:
        answer += "\n(No sources found)"

    await cl.Message(content=answer).send()
