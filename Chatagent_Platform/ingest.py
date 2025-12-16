# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# DATA_PATH = "data/"
# DB_FAISS_PATH = "vectorstores/db_faiss"


# # create vector DB
# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH, glob='*pdf', loader_cls=PyPDFLoader)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(
#         model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)


# if __name__ == '__main__':
#     create_vector_db()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os

DATA_PATH = "data/"
TXT_PATH = "cleaned_output.txt"
DB_FAISS_PATH = "vectorstores/db_faiss"


def create_vector_db():
    # --- Load PDF documents ---
    loader = DirectoryLoader(DATA_PATH, glob='*pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # --- Load cleaned .txt file ---
    if os.path.isfile(TXT_PATH):
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            txt_content = f.read()
            documents.append(Document(page_content=txt_content, metadata={"source": os.path.basename(TXT_PATH)}))
        print(f"✅ Loaded cleaned text: {TXT_PATH}")
    else:
        print(f"⚠️ Skipped: {TXT_PATH} not found")

    # --- Split documents into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # --- Embedding model ---
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    # --- Create and save FAISS index ---
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"📦 FAISS vectorstore created at: {DB_FAISS_PATH}")


if __name__ == '__main__':
    create_vector_db()

