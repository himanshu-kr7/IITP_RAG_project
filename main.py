import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables (your OPENAI_API_KEY)
load_dotenv()

# Define the path for the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_rag_chain():
    """
    Creates a Conversational Retrieval (RAG) chain with memory.
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

    # 5. Create a memory buffer
    # 'chat_history' is the key LangChain uses to pass history.
    # 'return_messages=True' ensures we get a list of messages.
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer' # Specify 'answer' as the output key
    )

    # 6. Create the ConversationalRetrievalChain
    # This chain does all the hard work:
    # - Takes the new question
    # - Looks at the chat_history
    # - Condenses them into a new, standalone question
    # - Retrieves documents based on that new question
    # - Generates an answer
    # - It also automatically returns the source documents!
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True, # This makes it return the sources
        output_key='answer' # Specify 'answer' as the output key
    )

    return chain

def main():
    """
    Main function to run the RAG query system.
    """
    # Create the RAG chain
    chain = create_rag_chain()

    print("--------------------------------------------------")
    print("IIT Patna AI Document Query System (with Memory)")
    print("Ask a question about your documents. Type 'exit' to quit.")
    print("--------------------------------------------------")

    # We don't need to manually manage chat_history here,
    # the 'memory' object inside the chain does it for us.

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

            # We just pass the new question.
            # The chain will automatically pull from its internal memory.
            # The input must be a dictionary with a 'question' key.
            response = chain.invoke({"question": query})

            print("\n--- Answer ---")
            print(response['answer'])

            print("\n--- Sources ---")
            # The sources are now in a key called 'source_documents'
            for doc in response['source_documents']:
                # Extract metadata
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')

                # Print the source and page number
                print(f"  - Source: {source},  Page: {page + 1}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()