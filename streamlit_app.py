import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from ddgs import DDGS
from hybrid_retriever import HybridRetriever
from query_rewriter import rewrite_query

load_dotenv()

st.set_page_config(
    page_title="SteelMind — Industrial RAG Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Page background ---- */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12151f 0%, #1c2033 100%);
    border-right: 1px solid #2a3050;
}
[data-testid="stSidebar"] * { color: #c8d0e8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8eaf6 !important; }

/* ---- Hero header ---- */
.hero {
    background: linear-gradient(135deg, #1e2a4a 0%, #2d3561 50%, #1a2340 100%);
    border: 1px solid #3d4f8a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
}
.hero-sub {
    font-size: 0.95rem;
    color: #b0bcd8;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    border: 1px solid rgba(99,102,241,0.5);
    color: #c4cafe;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem 1.25rem !important;
}

/* ---- Source pills ---- */
.source-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin: 0.15rem 0.2rem 0.15rem 0;
}
.source-pill-web {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.3);
    color: #6ee7b7;
}

/* ---- Rewrite badge ---- */
.rewrite-box {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid rgba(245,158,11,0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.4rem 0.8rem;
    margin: 0.5rem 0 0.75rem 0;
    font-size: 0.8rem;
    color: #fcd34d;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    background: #1a1f2e !important;
    border: 1px solid #2a3050 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ---- Sliders & toggles tint ---- */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #a5b4fc !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3561; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ---- Divider color ---- */
hr { border-color: #2a3050 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

_vectorstore = load_vectorstore()


def get_relevant_docs(question, threshold=1.0, top_k=5):
    retriever = HybridRetriever(_vectorstore, k=top_k)
    return retriever.retrieve(question, threshold=threshold)


def web_search(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def short_source(path: str) -> str:
    """Return just the filename without path or extension."""
    return os.path.splitext(os.path.basename(path))[0]


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "doc_hits" not in st.session_state:
    st.session_state.doc_hits = 0
if "web_hits" not in st.session_state:
    st.session_state.web_hits = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SteelMind")
    st.markdown("*Industrial Knowledge Assistant*")
    st.divider()

    st.markdown("### Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (K)", 1, 10, 5, 1,
        help="More chunks = richer context, but slower responses.")
    relevance_threshold = st.slider("Relevance threshold", 0.5, 2.0, 1.0, 0.1,
        help="Lower = stricter match required before using documents.")
    use_rewriter = st.toggle("Query rewriting", value=True,
        help="Expands your question into a keyword-rich search query.")

    st.divider()
    st.markdown("### Session Stats")
    st.markdown(f"**{st.session_state.total_queries}** questions asked")
    st.markdown(f"📄 **{st.session_state.doc_hits}** answered from docs")
    st.markdown(f"🌐 **{st.session_state.web_hits}** answered from web")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.doc_hits = 0
        st.session_state.web_hits = 0
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#4a5568;'>Powered by Ollama · FAISS · BM25<br>Model: mistral · Embeddings: MiniLM</div>",
        unsafe_allow_html=True,
    )


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Steel Manufacturing Intelligence</div>
  <h1 class="hero-title">SteelMind RAG Assistant</h1>
  <p class="hero-sub">Ask anything about steelmaking processes, equipment, quality standards, and defect analysis.<br>
  Answers are grounded in your technical document library — with live web fallback.</p>
</div>
""", unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten"):
            st.markdown(
                f'<div class="rewrite-box">✍️ Searched as: <em>{msg["rewritten"]}</em></div>',
                unsafe_allow_html=True,
            )
        if msg.get("doc_sources"):
            pills = "".join(
                f'<span class="source-pill">📄 {short_source(s)}</span>'
                for s in msg["doc_sources"]
            )
            st.markdown(f'<div style="margin-top:0.5rem">{pills}</div>', unsafe_allow_html=True)
        if msg.get("web_sources"):
            pills = "".join(
                f'<a href="{u}" target="_blank" class="source-pill source-pill-web">🌐 {u[:45]}…</a>'
                for u in msg["web_sources"][:3]
            )
            st.markdown(f'<div style="margin-top:0.5rem">{pills}</div>', unsafe_allow_html=True)


# ── Input & response ──────────────────────────────────────────────────────────
if question := st.chat_input("Ask about steelmaking, equipment, defects, quality standards…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        rewritten = None
        doc_sources = []
        web_sources = []
        source_used = None

        with st.spinner("Searching knowledge base…"):
            search_query = rewrite_query(question) if use_rewriter else question
            if use_rewriter and search_query != question:
                rewritten = search_query
                st.markdown(
                    f'<div class="rewrite-box">✍️ Searched as: <em>{search_query}</em></div>',
                    unsafe_allow_html=True,
                )

            docs = get_relevant_docs(search_query, threshold=relevance_threshold, top_k=top_k)
            context_parts = []

            if docs:
                context_parts.append("\n\n".join(d.page_content for d in docs))
                doc_sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
                source_used = "documents"
                st.session_state.doc_hits += 1
            else:
                st.toast("Not found in documents — searching the web…", icon="🌐")
                web_results = web_search(question)
                if web_results:
                    context_parts.append("\n\n".join(f"{r['title']}: {r['body']}" for r in web_results))
                    web_sources = [r.get("href", "") for r in web_results if r.get("href")]
                    source_used = "web"
                    st.session_state.web_hits += 1

        context = "\n\n".join(context_parts)

        recent_history = st.session_state.messages[-4:] if len(st.session_state.messages) > 1 else []
        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent_history)
        history_section = f"\nConversation so far:\n{history_text}\n" if history_text else ""

        prompt = f"""Answer the question based on the context below.{history_section}
Context:
{context}

Question: {question}

Answer:"""

        if not context:
            answer = "I couldn't find relevant information in the documents or on the web for this question."
        else:
            with st.spinner("Generating answer…"):
                llm = OllamaLLM(model="mistral")
                answer = llm.invoke(prompt)

        st.markdown(answer)

        # Source pills
        if doc_sources:
            pills = "".join(
                f'<span class="source-pill">📄 {short_source(s)}</span>' for s in doc_sources
            )
            st.markdown(f'<div style="margin-top:0.75rem">{pills}</div>', unsafe_allow_html=True)
        if web_sources:
            pills = "".join(
                f'<a href="{u}" target="_blank" class="source-pill source-pill-web">🌐 {u[:45]}…</a>'
                for u in web_sources[:3]
            )
            st.markdown(f'<div style="margin-top:0.75rem">{pills}</div>', unsafe_allow_html=True)

        st.session_state.total_queries += 1

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "rewritten": rewritten,
        "doc_sources": doc_sources,
        "web_sources": web_sources,
    })
