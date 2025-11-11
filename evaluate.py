import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from datasets import Dataset # We will use this to format our data
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# --- 1. SETUP ENVIRONMENT ---
print("Setting up environment...")
load_dotenv()

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY not found in .env file.")
    exit()

# Configuration
DATA_PATH = "data"
TEST_SET_FILE = "test_set.json" # Where we will save our questions

# --- 2. LOAD DOCUMENTS ---
print(f"Loading documents from {DATA_PATH}...")
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

if not documents:
    print(f"No PDF documents found in {DATA_PATH}. Please add files to evaluate.")
    exit()

print(f"Loaded {len(documents)} documents.")

# --- 3. GENERATE TEST SET (THE SLOW PART) ---
# This part will read your documents and use an LLM
# to create Questions, "ground_truth" answers, and contexts.
print("Generating test set... This may take several minutes...")

# Initialize the RAGAs test set generator
generator = TestsetGenerator.with_openai()

# We will generate a small set of 10 test questions.
# We'll use simple, reasoning, and multi-context questions.
test_set = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

# Convert to a dataset and save
print("Saving test set...")
test_set_df = test_set.to_pandas()
test_set_df.to_json(TEST_SET_FILE, orient="records", lines=True)

print("\n--- Test Set Generation Complete ---")
print(f"Your test set with {len(test_set_df)} questions has been saved to '{TEST_SET_FILE}'.")
print("You can now inspect this file.")