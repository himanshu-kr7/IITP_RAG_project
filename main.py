import streamlit as st
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- APPLICATION SETUP ---

# Load environment variables (ensure .env file is present)
from dotenv import load_dotenv
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with Your Documents", page_icon="📄")

# Set the title for the app
st.title("Chat with Your Documents 📄")

# --- GLOBAL VARIABLES & CACHING ---

# Define the paths
VECTOR_STORE_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"

# Cache the vector store and retriever to avoid reloading on every interaction
@st.cache_resource
def load_retriever():
    if not os.path.exists(VECTOR_STORE_PATH):
        st.error(f"Vector store not found at {VECTOR_STORE_PATH}. Please run `ingest.py` first.")
        return None

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Cache the conversational RAG chain
@st.cache_resource
def get_conversational_rag_chain(_retriever):
    # We use a simple ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Specify 'answer' as the output key
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_retriever,
        memory=memory,
        return_source_documents=True, # We'll want the sources
        output_key='answer' # Specify 'answer' as the output key
    )
    return chain

# Load the components
retriever = load_retriever()
if retriever:
    rag_chain = get_conversational_rag_chain(retriever)
else:
    rag_chain = None

# --- STREAMLIT CHAT INTERFACE ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):

    if not rag_chain:
        st.error("The application is not initialized. Please run `ingest.py`.")
    else:
        # 1. Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get the bot's response
        # We pass the prompt (question) and the existing chat history
        with st.spinner("Thinking..."):
            history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[:-1]]

            # Call the chain
            result = rag_chain.invoke({
                "question": prompt,
                "chat_history": history
            })

            answer = result["answer"]
            sources = result["source_documents"]

            # 3. Add bot response to session state and display it
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

                # Optionally display sources
                with st.expander("Show Sources"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(f"Content: {doc.page_content[:150]}...")