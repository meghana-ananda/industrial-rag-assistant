FROM python:3.11-slim

WORKDIR /app

# System deps for FAISS and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs as a sidecar — point OLLAMA_HOST to it if needed
ENV OLLAMA_HOST=http://ollama:11434

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
