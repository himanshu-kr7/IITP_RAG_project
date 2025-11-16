# Intelligent Document Query System: A Data-Driven RAG Project

**Author:** Himanshu (GitHub: [himanshu-kr7](https://github.com/himanshu-kr7))

## Project Overview

This project implements an advanced **Retrieval-Augmented Generation (RAG)** system designed to answer complex queries based *only* on a private knowledge base of PDF documents. Unlike general-purpose Large Language Models (LLMs) that can "hallucinate" or lack up-to-date information, this system ensures all generated answers are factual, verifiable, and directly sourced from the provided documents.

The core objective was not just to build a RAG application, but to **scientifically evaluate and iteratively improve its performance** using a data-driven approach.

## The Problem Solved

Standard LLMs face two critical challenges when applied to specific, domain-specific tasks:
1.  **Knowledge Cutoff:** They cannot access information beyond their training data, making them unsuitable for private or recent organizational documents.
2.  **Hallucination:** When faced with questions outside their knowledge base, LLMs tend to "make up" plausible but incorrect answers, undermining trust and reliability.

Our RAG system directly addresses these by grounding the LLM's responses in explicit, retrieved evidence from our document library.

## Final V1.3 Architecture: A 3-Stage Pipeline

Our project evolved from a simple "naive" RAG into a robust, multi-stage pipeline, demonstrating significant performance gains through iterative refinement.

1.  **Stage 1: Hybrid Retrieval**
    * Combines **BM25 (Keyword Search)** for precise term matching with **FAISS (Vector Search)** for semantic similarity.
    * This ensures comprehensive initial document retrieval, collecting 10 potentially relevant document chunks.
2.  **Stage 2: Contextual Re-Ranking**
    * Utilizes **Cohere's Re-ranker (`rerank-english-v3.0`)** to intelligently score and select the **Top 3 most relevant** document chunks from the initial 10. This crucial step filters out irrelevant context ("context pollution").
3.  **Stage 3: Grounded Generation with Smart Prompting**
    * The filtered Top 3 chunks are passed to a **`gpt-4o-mini`** LLM.
    * A custom **"Synthesis Prompt"** instructs the LLM to provide a helpful, coherent answer *derived solely from the provided context*. This prompt was a key breakthrough in improving answer quality.

## Data-Driven Performance Improvement

A cornerstone of this project was its rigorous, data-driven evaluation process using the **RAGAs** library. We tracked key metrics over multiple iterations, culminating in an **A+ score** for answer relevancy.

| RAG Strategy | Faithfulness (Answer Grounded in Docs) | Answer Relevancy (Answer Addresses Question) |
| :--- | :--- | :--- |
| **V1.0: Baseline (FAISS only)** | 61.4% | 47.7% (Fail) |
| **V1.1: + Cohere Re-Ranker** | 50.6% | 46.9% (Fail) |
| **V1.2: + Hybrid Search** | 45.7% | 46.8% (Fail) |
| **V1.3: Hybrid Search + Smart Prompt** | **78.8% (Good)** | **94.7% (A+ Success!)** |

**Key Learning:** Initial attempts to add sophisticated retrieval/ranking techniques *lowered* scores due to an overly strict LLM prompt. The significant jump in `Answer Relevancy` to **94.7%** was achieved primarily through **Prompt Engineering**, which "un-handcuffed" the LLM and allowed it to synthesize answers effectively from the provided context.

## Demo and Known Limitations

The application is deployed via Streamlit, offering an intuitive chat interface.

A notable limitation identified through rigorous testing, specifically with the query "requirement of admission in btech", highlighted the "Garbage In, Garbage Out" principle. While the RAG pipeline successfully retrieved the `B.Tech_Ordinance.pdf`, the document itself did **not** contain the requested admission requirements. This confirmed that even the most advanced RAG cannot answer questions if the information is genuinely missing from its source data.

## Technologies Used

* **Python 3.10+**
* **LangChain:** Core RAG framework, LCEL
* **Streamlit:** Web UI
* **OpenAI:** Embeddings (`text-embedding-ada-002`), LLM (`gpt-4o-mini`)
* **FAISS:** Vector Store
* **BM25:** Keyword Search (via `rank_bm25`)
* **Cohere:** Re-ranker (`rerank-english-v3.0`)
* **RAGAs:** Evaluation framework for generating test sets and scoring.

## How to Run This Project

To get a local copy up and running, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/himanshu-kr7/IITP_RAG_project.git](https://github.com/himanshu-kr7/IITP_RAG_project.git)
    cd IITP_RAG_project
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configure API Keys:**
    * Create a `.env` file in the root directory.
    * Add your API keys:
        ```
        OPENAI_API_KEY="your_openai_key_here"
        COHERE_API_KEY="your_cohere_key_here"
        ```
4.  **Prepare Your Data:**
    * Create a `data/` directory in the root of the project.
    * Place your `.pdf` documents inside this `data/` folder.
    * *(Note: Source data is not provided in this repository due to copyright; use your own PDFs.)*
5.  **Ingest Documents:**
    ```bash
    python ingest.py
    ```
    This will create the `vectorstore/` directory with your embeddings.
6.  **Run the Streamlit App:**
    ```bash
    python -m streamlit run main.py
    ```
    The application will open in your web browser.

---