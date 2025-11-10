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
    This chain will return a dictionary with 'answer' and 'context' (sources).
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

    """
    Creates the Retrieval-Augmented Generation (RAG) chain using LCEL.
    This chain will return a dictionary with 'answer' and 'context' (sources).
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
    retriever = db.as_retriever(search_kwargs={'k': 3})

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

    # This chain will take the input dictionary {"question": ...}
    # and add a new key, "context", containing the retrieved documents.
    context_retriever_chain = RunnablePassthrough.assign(
        context=(lambda x: retriever.invoke(x["question"]))
    )

    # This chain is a RunnableParallel map.
    # It takes the dict, formats the context, and passes the question.
    # The output is a dict that can be passed to the prompt.
    answer_generation_chain = (
        {"context": (lambda x: format_docs(x["context"])), "question": (lambda x: x["question"])}
        | prompt
        | llm
        | StrOutputParser()
    )

    # This is the final chain.
    # 1. It runs context_retriever_chain, which gets the context.
    # 2. It pipes that output dict to the next .assign()
    # 3. The new .assign() runs answer_generation_chain to get the answer
    # 4. It passes through the 'context' key and adds the 'answer' key
    rag_chain_with_sources = (
        context_retriever_chain |
        RunnablePassthrough.assign(answer=answer_generation_chain)
    )

    return rag_chain_with_sources

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

            # We must pass a dictionary and will get a dictionary back
            response = chain.invoke({"question": query})

            print("\n--- Answer ---")
            print(response['answer'])

            print("\n--- Sources ---")
            # Loop through the 'context' (which is a list of Document objects)
            for doc in response['context']:
                # Extract metadata
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')

                # Print the source and page number
                # We add 1 to the page number because they are 0-indexed
                print(f"  - Source: {source},  Page: {page + 1}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()