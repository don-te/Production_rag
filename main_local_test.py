# main_local_test.py - Script to test the modular RAG model

import os
from dotenv import load_dotenv
import sys

# Import the core RAG logic from our module
from rag_model import initialize_rag_chain, LLM_MODEL_NAME

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not found in .env file.")
        print("Please ensure it is set correctly. Exiting.")
        sys.exit(1)

    print(f"--- Initializing RAG Chain with Model: {LLM_MODEL_NAME} ---")
    
    try:
        # Call the external module's function to get the runnable chain
        rag_chain, model_name = initialize_rag_chain(OPENROUTER_API_KEY)
        print("--- Initialization Complete. Starting test loop. ---")
    except Exception as e:
        print(f"\nFATAL ERROR during model initialization:")
        print(f"  Details: {e}")
        print("Check your LangChain, Chroma, and LiteLLM installations, and API Key.")
        sys.exit(1)

    # ----------------------------------------------------
    # Test Loop
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("Testing RAG chain. Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question about FastAPI: ")
        if query.lower() == 'exit':
            break
        
        try:
            # Invoke the chain
            answer = rag_chain.invoke(query)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"\n[QUERY ERROR] Could not get an answer. Check LangSmith for trace: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()