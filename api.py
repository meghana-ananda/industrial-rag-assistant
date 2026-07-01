"""
FastAPI REST API for the Industrial RAG Assistant.

Endpoints:
  POST /query   — ask a question, get an answer grounded in documents
  GET  /health  — liveness check

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from hybrid_retriever import HybridRetriever
from query_rewriter import rewrite_query


# --- shared state loaded once at startup ---
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    _state["retriever"] = HybridRetriever(vs, k=5)
    _state["llm"] = OllamaLLM(model="mistral")
    yield
    _state.clear()


app = FastAPI(
    title="Industrial RAG Assistant",
    description="Ask questions about steel manufacturing — answers grounded in technical documents.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- request / response schemas ---

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float = 1.0
    rewrite: bool = True


class QueryResponse(BaseModel):
    question: str
    rewritten_query: str | None
    answer: str
    sources: list[str]
    num_chunks_used: int


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    retriever: HybridRetriever = _state["retriever"]
    llm: OllamaLLM = _state["llm"]

    rewritten = rewrite_query(req.question) if req.rewrite else req.question
    retriever.k = req.top_k
    docs = retriever.retrieve(rewritten, threshold=req.threshold)

    if not docs:
        return QueryResponse(
            question=req.question,
            rewritten_query=rewritten if rewritten != req.question else None,
            answer="I couldn't find relevant information in the documents for this question.",
            sources=[],
            num_chunks_used=0,
        )

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

    prompt = f"""Answer the question based ONLY on the context below. Do not use your own knowledge.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {req.question}

Answer:"""

    answer = llm.invoke(prompt)

    return QueryResponse(
        question=req.question,
        rewritten_query=rewritten if rewritten != req.question else None,
        answer=answer,
        sources=sorted(sources),
        num_chunks_used=len(docs),
    )
