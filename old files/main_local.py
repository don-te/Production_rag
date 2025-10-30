# main_local.py (CORRECTED IMPORTS)
import os
from dotenv import load_dotenv

# --- RAG Libraries ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# *** THE FIX IS HERE ***
# ChatPromptTemplate and other core items are now in langchain_core
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ... rest of your code ...


# Load environment variables
load_dotenv()

# --- Define models ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/gemma-3n-e2b-it:free" 

def main():
    # 1. Initialize the LLM and Embedding Model
    # --- FIXED: Use api_key parameter and ensure correct base_url ---
    llm = ChatOpenAI(
        # Use the correct API key parameter name
        api_key=os.getenv("OPENROUTER_API_KEY"),
        
        # Use OpenRouter's API base URL
        base_url="https://openrouter.ai/api/v1",
        
        # Pass the model name
        model=LLM_MODEL_NAME,
        temperature=0.1
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    # ---------------

    # 2. Load the vector store
    # Ensure the chroma_db exists (run the ingestion script once)
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Create the RAG Chain
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Test the chain
    print(f"Testing RAG chain with OpenRouter ({LLM_MODEL_NAME}). Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question about FastAPI: ")
        if query.lower() == 'exit':
            break
        
        # Invoke the chain
        answer = rag_chain.invoke(query)
        print("Answer:", answer)
        print("-" * 20)

if __name__ == "__main__":
    main()