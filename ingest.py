import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Define the path to your data directory
DATA_PATH = "data/"
# Define the path for the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    Creates a vector database from PDF documents in the DATA_PATH.
    
    1. Loads PDFs from the specified directory.
    2. Splits the documents into smaller text chunks.
    3. Generates embeddings for these chunks.
    4. Saves the embeddings in a FAISS vector store.
    """
    
    # 1. Load PDF documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found in the data directory.")
        return
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 3. Generate embeddings
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create and save the FAISS vector store
    print("Creating and saving the vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved successfully at {DB_FAISS_PATH}!")

if __name__ == "__main__":
    create_vector_db()