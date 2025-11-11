import os
import shutil
# --- ADD THESE TWO LINES ---
from dotenv import load_dotenv
load_dotenv() # This loads the .env file
# ---------------------------
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- CONFIGURATION ---
DATA_PATH = "data"
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# --- 1. LOAD DOCUMENTS ---
print("Loading PDF documents...")

# Use the stable PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

if not documents:
    print("No PDF documents found in the 'data' directory. Please add your PDF files.")
    exit()

print(f"Loaded {len(documents)} PDF documents.")

# --- 2. SPLIT DOCUMENTS ---
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunks)} chunks.")

# --- 3. CREATE EMBEDDINGS & VECTOR STORE ---
print("Creating embeddings and vector store...")

# Use OpenAIEmbeddings, just like in main.py
embeddings_model = OpenAIEmbeddings()

# Check if old vector store exists and remove it
if os.path.exists(VECTOR_STORE_PATH):
    print(f"Removing old vector store at {VECTOR_STORE_PATH}...")
    shutil.rmtree(VECTOR_STORE_PATH)

# Create a new vector store from the chunks
vector_store = FAISS.from_documents(chunks, embeddings_model)

# --- 4. SAVE VECTOR STORE ---
print(f"Saving vector store locally at {VECTOR_STORE_PATH}...")
vector_store.save_local(VECTOR_STORE_PATH)

print("\n--- Ingestion Complete ---")
print(f"Vector store is saved and ready at '{VECTOR_STORE_PATH}'.")