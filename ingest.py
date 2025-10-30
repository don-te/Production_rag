# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <-- CHANGED
from langchain_community.vectorstores import Chroma
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# --- NEW: Define which embedding model to use ---
# We'll use a small, fast, and popular model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # 1. Define the data source
    loader = WebBaseLoader("https://fastapi.tiangolo.com/tutorial/first-steps/")
    docs = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create the embedding model
    #    This will run locally on your machine
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # --- CHANGED ---
    # We initialize the HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Force running on CPU
    )
    # ---------------

    # 4. Create the vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,  # <-- Use our new local embeddings
        persist_directory="./chroma_db"
    )
    
    print("Data ingestion complete. Vector store created in './chroma_db'")

if __name__ == "__main__":
    main()