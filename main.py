import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (your OPENAI_API_KEY)
load_dotenv()

# Define the path for the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_rag_chain():
    """
    Creates the Retrieval-Augmented Generation (RAG) chain using LCEL.
    """

    # 1. Initialize the LLM (Using OpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    # 2. Load the embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Load the FAISS vector store
    print("Loading vector store...")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded.")

    # 4. Create a retriever
    retriever = db.as_retriever(search_kwargs={'k': 3}) # 'k': 3 means it will retrieve the top 3 relevant chunks

    # 5. Define the RAG prompt manually
    RAG_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 6. Create the RAG chain using LCEL

    def format_docs(docs):
        # Joins the retrieved documents into a single string
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    """
    Main function to run the RAG query system.
    """
    # Create the RAG chain
    chain = create_rag_chain()

    print("--------------------------------------------------")
    print("IIT Patna AI Document Query System (OpenAI STABLE)")
    print("Ask a question about your documents. Type 'exit' to quit.")
    print("--------------------------------------------------")

    while True:
        # Get user input
        query = input("\nAsk your question: ")

        if query.lower() == 'exit':
            print("Exiting...")
            break

        if not query.strip():
            continue

        # Get the answer from the chain
        try:
            print("Thinking...")

            answer = chain.invoke(query)

            print("\n--- Answer ---")
            print(answer)

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()