import os
import shutil
from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
DATA_PATH = "data"
VECTOR_STORE_PATH = "vectorstore/db_faiss"

def main():
    """
    This script loads documents from the DATA_PATH, splits them into chunks,
    creates OpenAI embeddings, and saves them to a FAISS vector store.
    """
    print("Starting data ingestion process...")

    # 1. Load documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print(f"No PDF documents found in {DATA_PATH}. Aborting.")
        return
    print(f"Loaded {len(documents)} PDF documents.")

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Create embeddings and vector store
    print("Creating OpenAI embeddings and FAISS vector store...")
    embeddings_model = OpenAIEmbeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Removing old vector store at {VECTOR_STORE_PATH}...")
        shutil.rmtree(VECTOR_STORE_PATH)

    vector_store = FAISS.from_documents(chunks, embeddings_model)

    # 4. Save vector store
    print(f"Saving vector store locally at {VECTOR_STORE_PATH}...")
    vector_store.save_local(VECTOR_STORE_PATH)

    print("\n--- Ingestion Complete ---")
    print(f"Vector store is saved and ready at '{VECTOR_STORE_PATH}'.")

if __name__ == "__main__":
    main()