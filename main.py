import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter

# --- Core Imports ---
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Application Setup ---
load_dotenv()

st.set_page_config(page_title="Chat with Your Documents", page_icon="📄")
st.title("Chat with Your Documents 📄")

VECTOR_STORE_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"

# --- Model Loading and RAG Chain ---

@st.cache_resource
def load_models():
    """
    Load all documents, create two retrievers (BM25 and FAISS),
    combine them into an EnsembleRetriever, and wrap them with a Re-ranker.
    This function is cached by Streamlit for performance.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        st.error(f"Vector store not found at {VECTOR_STORE_PATH}. Please run `ingest.py` first.")
        return None, None

    # 1. Load documents from disk (for BM25)
    loader = PyPDFDirectoryLoader(DATA_PATH)
    all_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = text_splitter.split_documents(all_docs)
    
    # 2. Initialize Keyword Retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 5

    # 3. Initialize Vector Retriever (FAISS)
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
    
    return compression_retriever, llm

def format_docs(docs):
    """Format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    """Format chat history into a string for the prompt."""
    buffer = []
    for message in chat_history:
        buffer.append(f"{message['role'].capitalize()}: {message['content']}")
    return "\n".join(buffer)

# The "synthesis" prompt for smart answering
prompt_template = """
You are an expert assistant for answering questions about the provided documents.
Use the following pieces of retrieved context to answer the user's question.
Synthesize a helpful, coherent answer. Try to use information from all relevant sources.
If the answer is truly not in the context, *then* say "I don't have that information in the documents."

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Load the models
retriever, llm = load_models()

# Define the RAG chain
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
            "context": itemgetter("documents")
        }
    )
else:
    rag_chain = None

# --- Streamlit Chat UI ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask a question about your documents..."):
    
    if rag_chain:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.spinner("Thinking... (hybrid search & re-ranking)"):
            history_string = format_chat_history(st.session_state.messages[:-1])
            
            result = rag_chain.invoke({
                "question": prompt_input,
                "chat_history": history_string
            })
            
            answer = result["answer"]
            sources = result["context"] 

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                with st.expander("Show Sources"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(f"Content: {doc.page_content[:150]}...")
    else:
        st.error("Application is not initialized. Please run `ingest.py` and restart.")