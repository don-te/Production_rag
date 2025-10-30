# Dockerfile (Production Version)

# Stage 1: Build Stage - Install dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace Embedding Model
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Stage 2: Final Image
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application code and necessary assets
COPY api.py .
COPY rag_model.py .
COPY chroma_db ./chroma_db/ 
# --- NOTICE THE .env FILE IS NO LONGER COPIED ---

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]