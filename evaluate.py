import os
import json
import time # For Cohere rate limiting
from dotenv import load_dotenv
from datasets import Dataset
from tqdm import tqdm 
from functools import lru_cache

# --- RAGAs Imports ---
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper

# --- LangChain Imports ---
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Hybrid Search Imports ---
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# --- 1. Configuration ---
load_dotenv()
VECTOR_STORE_PATH = "vectorstore/db_faiss"
TEST_SET_FILE = "test_set.json"
DATA_PATH = "data"

# --- 2. Build the RAG Chain (Mirrors main.py) ---

@lru_cache(maxsize=None) 
def load_models():
    """
    Loads and initializes all components for the RAG pipeline.
    This includes the LLM, Embedder, and the full Hybrid Search retriever.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}.")

    # 1. Load documents from disk (for BM25)
    print("Loading documents for BM25 index...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    all_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = text_splitter.split_documents(all_docs)
    
    # 2. Initialize Keyword Retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 5

    # 3. Initialize Vector Retriever (FAISS)
    print("Loading FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    # 4. Initialize Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5] 
    )
    
    # 5. Initialize Cohere Re-ranker
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=3) 
    
    # 6. Create the final Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=ensemble_retriever
    )
    
    # 7. Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    print("All models loaded successfully.")
    return compression_retriever, llm

def format_docs(docs):
    """Format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Define the "synthesis" prompt
prompt_template = """
You are an expert assistant for answering questions about the provided documents.
Use the following pieces of retrieved context to answer the user's question.
Synthesize a helpful, coherent answer. Try to use information from all relevant sources.
If the answer is truly not in the context, *then* say "I don't have that information in the documents."

Context:
{context}
Chat History: {chat_history}
User Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Load the models
retriever, llm = load_models()

# Define the full RAG chain
if retriever and llm:
    context_chain = (
        RunnableParallel({
            "documents": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        })
        | RunnableParallel({
            "context": itemgetter("documents") | RunnableLambda(format_docs),
            "documents": itemgetter("documents"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        })
    )
    rag_chain = (
        context_chain
        | {
            "answer": prompt | llm | StrOutputParser(),
            "contexts": itemgetter("documents") 
        }
    )
else:
    rag_chain = None

def run_evaluation():
    """
    Main function to run the RAG evaluation.
    """
    if not rag_chain:
        print("ERROR: RAG chain not loaded. Cannot run evaluation.")
        return

    # --- 3. LOAD THE TEST SET ---
    print(f"Loading test set from {TEST_SET_FILE}...")
    test_data = []
    with open(TEST_SET_FILE, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    if not test_data:
        print(f"Error: {TEST_SET_FILE} is empty or not found.")
        return
    print(f"Loaded {len(test_data)} test questions.")

    # --- 4. RUN RAG CHAIN ON ALL TEST QUESTIONS ---
    print("Running RAG chain on all test questions to get answers (with rate limit)...")
    results = []
    for item in tqdm(test_data):
        question = item['user_input'] 
        chat_history = "" 
        
        result = rag_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        results.append({
            "question": question,
            "answer": result['answer'],
            "contexts": [doc.page_content for doc in result['contexts']],
        })
        
        # Wait 6 seconds to stay under the Cohere free tier limit (10 calls/minute)
        time.sleep(6)

    print("Finished generating answers.")

    # --- 5. PREPARE DATASET FOR RAGAS ---
    eval_dataset = Dataset.from_list(results)

    # --- 6. RUN RAGAS EVALUATION ---
    print("Running RAGAs evaluation... This may take a few minutes...")
    metrics = [
        faithfulness,
        answer_relevancy,
    ]
    ragas_llm = LangchainLLMWrapper(llm)

    score = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=OpenAIEmbeddings()
    )

    # --- 7. SHOW THE RESULTS ---
    print("\n--- RAG Evaluation Complete (Hybrid Search + New Prompt) ---")
    print(score)

# --- Main execution ---
if __name__ == "__main__":
    run_evaluation()