# rag_model.py - RAG Core Logic Module

import os
from typing import Any, List, Mapping, Optional
from litellm import completion 

# --- LangChain Core Imports ---
from langchain_chroma import Chroma 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage 
from langchain_core.outputs import ChatResult, ChatGeneration 
from langchain_core.language_models import BaseChatModel

# --- Configuration (Define constants here) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openrouter/meta-llama/llama-3.3-8b-instruct:free" 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" # Constant URL

# =========================================================================
# Custom LLM Wrapper for LiteLLM
# =========================================================================

class LiteLLMChat(BaseChatModel):
    """A custom LangChain Chat Model wrapper for LiteLLM."""
    model_name: str = LLM_MODEL_NAME
    api_key: str # API key passed on instantiation (not stored as constant here)

    @property
    def _llm_type(self) -> str:
        return "litellm-chat"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any,) -> ChatResult: 
        litellm_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
            litellm_messages.append({"role": role, "content": message.content})

        try:
            response = completion(
                model=self.model_name,
                messages=litellm_messages,
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
                temperature=0.1,
                **kwargs
            )
            content = response.choices[0].message.content
            ai_message = AIMessage(content=content)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation]) 
        except Exception as e:
            # Re-raising the error is vital for LangSmith tracing to capture the failure
            raise e 

    def _call(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> str:
        raise NotImplementedError

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
        
# =========================================================================
# The Reusable Initialization Function
# =========================================================================

def initialize_rag_chain(openrouter_api_key: str):
    """
    Initializes and returns the complete RAG chain and the LLM model name.
    
    This is the only function 'api.py' needs to call.
    """
    
    # 1. Initialize LLM (Custom LiteLLM wrapper)
    llm = LiteLLMChat(api_key=openrouter_api_key)
    
    # 2. Initialize Embeddings and load vector store
    # The embeddings must be initialized here to load Chroma correctly
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Define the RAG Chain (LCEL)
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

    return rag_chain, LLM_MODEL_NAME # Return both the chain and the name