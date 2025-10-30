# api.py - FastAPI Server (Modularized)

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 

# 🌟 MODULARITY: Import the initialization function from our new module
from rag_model import initialize_rag_chain, LLM_MODEL_NAME # Also import the model name constant

# --- Configuration ---
load_dotenv()
# API Key is read ONCE from the environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

# --- FastAPI Setup ---
app = FastAPI(
    title="Modular AI RAG API",
    description="Core AI logic is separated into a reusable rag_model.py module."
)

# --- Pydantic Schemas ---
class QueryInput(BaseModel):
    question: str

class QueryOutput(BaseModel):
    answer: str

# Global variable to hold the RAG chain
rag_chain = None

# ----------------------------------------------------
# 🌟 MLOps: Initialization and Dependency Loading 🌟
# ----------------------------------------------------
@app.on_event("startup")
def startup_event():
    """Initializes the RAG chain by calling the external module."""
    global rag_chain
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not found.")

    print("--- Initializing AI (Importing from rag_model.py) ---")
    try:
        # Call the external module function
        rag_chain, model_name = initialize_rag_chain(OPENROUTER_API_KEY)
        print(f"--- Initialization Complete. Model: {model_name} ---")
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")


# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": f"RAG API is online. LLM: {LLM_MODEL_NAME}"}

@app.post("/rag/query", response_model=QueryOutput)
def run_rag_query(input_data: QueryInput):
    """Endpoint to run a query against the RAG system."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized.")
    
    try:
        # Invoke the chain (the complex logic is hidden inside the rag_model module)
        answer = rag_chain.invoke(input_data.question)
        return QueryOutput(answer=answer)
    except Exception as e:
        # Catch and report errors, essential for production
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")