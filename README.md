# Industrial RAG Assistant

A production-ready Retrieval Augmented Generation (RAG) system for querying steel manufacturing documentation. Ask natural language questions and get accurate answers grounded in technical PDFs.

**[🚀 Try the Live Demo](#deployment)** | [📖 Read the Docs](#setup--installation) | [💻 GitHub](https://github.com/meghana-ananda/industrial-rag-assistant)

---

## 🎯 Overview

Steel manufacturing plants store critical knowledge across thousands of pages of technical documentation. Engineers need fast, accurate answers when problems arise on the production floor.

This project demonstrates a **production-ready RAG pipeline** that:
- ✅ Ingests industrial PDFs from local files OR web URLs
- ✅ Makes documents semantically searchable using embeddings
- ✅ Retrieves relevant context and grounds LLM responses
- ✅ Returns answers with source citations
- ✅ Runs entirely locally (no cloud required) OR integrates with Gemini
- ✅ Deploys for free to Streamlit Cloud

**Why this matters:** Built by someone who has worked inside steel manufacturing (PSI Metals), this project reflects real-world pain points and domain expertise, not a generic demo.

---

## 🏗️ Architecture

```
┌─────────────────────────────┐
│  PDFs (Local + Web URLs)    │  963 pages of steel docs
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  PyPDFLoader                │  Extract text from PDFs
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  RecursiveCharacterTextSplitter  │  1000-char chunks
└──────────┬──────────────────┘  │  200-char overlap
           │
           ▼
┌─────────────────────────────┐
│  HuggingFace Embeddings     │  all-MiniLM-L6-v2
│  (Local, Free, Fast)        │  No API calls needed
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FAISS Vector Store         │  4,951 searchable chunks
└──────────┬──────────────────┘
           │
     ┌─────┴──────────────┐
     │                    │
     ▼                    ▼
┌──────────┐        ┌──────────────┐
│  Query   │        │  Retriever   │  Top-5 semantic search
└────┬─────┘        └────┬─────────┘
     │                   │
     └───────┬───────────┘
             │
             ▼
     ┌──────────────────┐
     │  Ollama/Gemini   │  Generate grounded answer
     │  (LLM)           │
     └────┬─────────────┘
          │
          ▼
     ┌──────────────────────┐
     │  Answer + Sources    │  Fully cited
     └──────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **PDF Loading** | PyPDFLoader + requests | Load from files or URLs |
| **Chunking** | RecursiveCharacterTextSplitter | Preserves document structure |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free, local, 25MB model |
| **Vector Store** | FAISS | Fast similarity search, production-grade |
| **LLM** | Ollama (local) + Gemini (cloud) | Flexible, cost-effective |
| **Web UI** | Streamlit | Python-native, easy deployment |
| **Framework** | LangChain | Abstracts complexity, integrates everything |

---

## 📁 Project Structure

```
industrial-rag-assistant/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not in GitHub)
├── .gitignore                   # Excludes venv, vectorstore, .env
│
├── ingest.py                    # Load PDFs from local/web + vectorize
├── query.py                     # Interactive query interface
├── streamlit_app.py             # Web UI (Streamlit)
│
├── industrial_rag/              # Local PDFs folder (optional)
│   └── *.pdf
│
└── vectorstore/                 # Auto-generated FAISS index
    ├── index.faiss
    ├── index.pkl
    └── docstore.pkl
```

---

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/meghana-ananda/industrial-rag-assistant.git
cd industrial-rag-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment (Optional)
Create `.env`:
```
GOOGLE_API_KEY=your-gemini-key-here
```
(Only needed if using Gemini instead of Ollama)

### 5. Download/Index PDFs

**Option A: Load from web URLs**
```python
# Edit ingest.py and add URLs:
steel_industry_urls = [
    "https://example.com/steel-standard.pdf",
    "https://example.com/process-guide.pdf",
]

# Then run:
python ingest.py
```

**Option B: Use local PDFs**
```bash
# Create folder and add your PDFs
mkdir industrial_rag
# Copy your PDFs into industrial_rag/

python ingest.py
```

**Option C: Mix local + web**
```python
# ingest.py handles both automatically
ingest_documents(docs_folder="industrial_rag", urls=steel_industry_urls)
```

### 6. Run Interactive Query
```bash
python query.py
```

Type questions:
```
Your question: What is basic oxygen steelmaking?
Answer: [grounded response with sources]
```

### 7. (Optional) Run Web UI
```bash
# First, make sure Ollama is running:
ollama serve

# In another terminal:
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Deploy in one click

**Note:** Streamlit Cloud runs on free tier, so it'll use Ollama. For production, upgrade to include Gemini API key.

### Deploy REST API (Optional)

```bash
pip install fastapi uvicorn
```

Create `api.py`:
```python
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = FastAPI()

@app.post("/query")
def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings, 
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = Ollama(model="mistral")
    answer = llm.invoke(f"Context: {context}\n\nQ: {question}")
    
    return {
        "question": question,
        "answer": answer,
        "sources": [d.metadata.get('source') for d in docs]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run:
```bash
python api.py
```

Test:
```bash
curl -X POST "http://localhost:8000/query?question=What is basic oxygen steelmaking?"
```

---

## 📖 Usage

### Interactive CLI
```bash
python query.py
```
- Type questions
- Type `quit` to exit

### Streamlit Web App
```bash
streamlit run streamlit_app.py
```
- Beautiful UI
- Adjustable retrieval parameters
- Example questions
- Source citations

### Python Code
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, 
                               allow_dangerous_deserialization=True)

# Retrieve + answer
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("Your question here")
llm = Ollama(model="mistral")
answer = llm.invoke(f"Context: {docs}\n\nQ: ...")
```

---

## 💡 Example Questions

- "What is basic oxygen steelmaking?"
- "What are the main steps in steel production?"
- "What defects can occur in steel?"
- "What equipment is used in rolling mills?"
- "What are the quality standards for steel?"
- "What causes surface defects in steel?"
- "How is steel hardened?"
- "What is the difference between hot and cold rolling?"

---

## 📊 How It Works

### Document Ingestion
1. Load PDFs from local files or web URLs
2. Extract text with PyPDFLoader
3. Split into chunks (1000 chars, 200-char overlap)
4. Generate embeddings using HuggingFace
5. Store in FAISS for fast retrieval

### Query Processing
1. User asks a question
2. Convert question to embedding
3. Find top-5 most similar chunks (semantic search)
4. Assemble context from chunks
5. Pass to LLM with system prompt
6. LLM generates answer grounded in context
7. Return answer + source citations

### Why RAG?
- ✅ No hallucinations (answers grounded in docs)
- ✅ Always traceable (sources cited)
- ✅ Works with proprietary/domain-specific knowledge
- ✅ Scales to large document collections

---

## ⚙️ Configuration

### Chunking Strategy
Edit `ingest.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Parameters
Edit `query.py` or `streamlit_app.py`:
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Number of chunks to retrieve
)
```

### LLM Selection
In `query.py`, choose:
```python
# Option 1: Local (free, no API needed)
llm = Ollama(model="mistral")

# Option 2: Cloud (requires API key)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```

---

## 🔧 Troubleshooting

### "Vectorstore not found"
Run ingestion first:
```bash
python ingest.py
```

### "Ollama not running"
```bash
ollama pull mistral
ollama serve
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "PDF download failed"
- Check URL is valid and accessible
- Some PDFs may require authentication
- Try downloading manually and using local folder

### Slow responses
- First run loads embedding model (~25MB, one-time)
- Ollama slower than cloud LLMs
- Increase `k` in retriever for more context

---

## 📈 Performance

**Current Setup:**
- Documents indexed: 8 PDFs, 963 pages
- Chunks created: 4,951
- Retrieval speed: ~0.5 seconds
- Answer latency: ~2-5 seconds (Ollama)

**Optimization Ideas:**
- Hybrid search (BM25 + semantic)
- Caching frequent queries
- Async processing
- GPU acceleration for embeddings

---

## 🎓 Interview Talking Points

**"Tell me about this project"**
- RAG system for grounding LLM answers in steel manufacturing docs
- Loads PDFs from local files OR web URLs
- Semantic search with FAISS retrieval
- Works locally with Ollama or cloud with Gemini
- Deployed as interactive web app

**"How would you evaluate it?"**
- Retrieval accuracy: Are top-5 chunks relevant?
- Answer quality: Grounded, no hallucinations?
- User satisfaction: Did answer solve the problem?
- Use RAGAS framework for systematic evaluation

**"What would you add?"**
- Hybrid BM25 + semantic search
- Fine-tuned embeddings for manufacturing terms
- Multilingual support (German, French)
- REST API for integration
- User feedback loop for continuous improvement

---

## 📚 Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

---

## 📄 License

MIT License — Use this as a template for your own RAG projects.

---

## 🤝 Contributing

Ideas for improvements:
- Hybrid retrieval (BM25 + semantic)
- Multilingual queries
- Fine-tuned embeddings
- Web UI enhancements
- REST API
- Evaluation metrics

Open an issue or submit a PR!

---

**Built with ❤️ for steel manufacturing engineers**

Last updated: June 2026