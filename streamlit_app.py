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
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide the sidebar toggle arrow entirely */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp {
    background: linear-gradient(160deg, #0d1117 0%, #161b27 60%, #0d1117 100%);
}

/* ---- Hero ---- */
.hero {
    background: linear-gradient(135deg, #1b2646 0%, #252f55 60%, #1a2040 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.2rem 2.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.45);
    color: #c4cafe;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 0.9rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 0 0.55rem 0 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
}
.hero-sub {
    font-size: 0.93rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.65;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem 1.3rem !important;
}

/* ---- Source pills ---- */
.source-pill {
    display: inline-block;
    background: rgba(99,102,241,0.13);
    border: 1px solid rgba(99,102,241,0.28);
    color: #a5b4fc;
    font-size: 0.71rem;
    font-weight: 500;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    margin: 0.15rem 0.2rem 0 0;
    text-decoration: none;
}
.source-pill-web {
    background: rgba(16,185,129,0.1);
    border-color: rgba(16,185,129,0.28);
    color: #6ee7b7;
}

/* ---- Rewrite badge ---- */
.rewrite-box {
    background: rgba(245,158,11,0.07);
    border-left: 3px solid rgba(245,158,11,0.45);
    border-radius: 0 8px 8px 0;
    padding: 0.35rem 0.8rem;
    margin: 0.5rem 0 0.7rem 0;
    font-size: 0.79rem;
    color: #fcd34d;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    background: #151b2b !important;
    border: 1px solid #2a3050 !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2d3561; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ---- General text ---- */
p, li, span { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

_vectorstore = load_vectorstore()

# Fixed defaults — no sidebar needed
TOP_K = 5
THRESHOLD = 1.0
USE_REWRITER = True


def get_relevant_docs(question):
    retriever = HybridRetriever(_vectorstore, k=TOP_K)
    return retriever.retrieve(question, threshold=THRESHOLD)


def web_search(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def short_source(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Steel Manufacturing Intelligence</div>
  <p class="hero-title">SteelMind RAG Assistant</p>
  <p class="hero-sub">
    Ask anything about steelmaking processes, equipment, quality standards, and defect analysis.<br>
    Answers are grounded in your technical document library — with live web fallback.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.clear-btn {
    position: fixed;
    bottom: 90px;
    right: 28px;
    z-index: 999;
    background: rgba(30,36,58,0.85);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50%;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    cursor: pointer;
    backdrop-filter: blur(6px);
    transition: background 0.2s, border-color 0.2s;
    text-decoration: none;
}
.clear-btn:hover {
    background: rgba(99,102,241,0.25);
    border-color: rgba(99,102,241,0.6);
}
</style>
""", unsafe_allow_html=True)

# Fixed bottom-right clear button via a hidden Streamlit button triggered by JS click
if st.button("🗑️", key="clear_btn", help="Clear chat", type="secondary"):
    st.session_state.messages = []
    st.rerun()

st.markdown("""
<style>
/* Move the clear button to fixed bottom-right */
div[data-testid="stButton"] > button[kind="secondary"] {
    position: fixed !important;
    bottom: 90px !important;
    right: 28px !important;
    z-index: 999 !important;
    width: 42px !important;
    height: 42px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    background: rgba(30,36,58,0.85) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    font-size: 1.1rem !important;
    backdrop-filter: blur(6px) !important;
    min-width: unset !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(99,102,241,0.6) !important;
}
</style>
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
                f'<a href="{u}" target="_blank" class="source-pill source-pill-web">🌐 {u[:50]}…</a>'
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

        with st.spinner("Searching knowledge base…"):
            search_query = rewrite_query(question) if USE_REWRITER else question
            if USE_REWRITER and search_query != question:
                rewritten = search_query
                st.markdown(
                    f'<div class="rewrite-box">✍️ Searched as: <em>{search_query}</em></div>',
                    unsafe_allow_html=True,
                )

            docs = get_relevant_docs(search_query)
            context_parts = []

            if docs:
                context_parts.append("\n\n".join(d.page_content for d in docs))
                doc_sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
            else:
                st.toast("Not found in documents — searching the web…", icon="🌐")
                web_results = web_search(question)
                if web_results:
                    context_parts.append("\n\n".join(f"{r['title']}: {r['body']}" for r in web_results))
                    web_sources = [r.get("href", "") for r in web_results if r.get("href")]

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

        if doc_sources:
            pills = "".join(f'<span class="source-pill">📄 {short_source(s)}</span>' for s in doc_sources)
            st.markdown(f'<div style="margin-top:0.75rem">{pills}</div>', unsafe_allow_html=True)
        if web_sources:
            pills = "".join(
                f'<a href="{u}" target="_blank" class="source-pill source-pill-web">🌐 {u[:50]}…</a>'
                for u in web_sources[:3]
            )
            st.markdown(f'<div style="margin-top:0.75rem">{pills}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "rewritten": rewritten,
        "doc_sources": doc_sources,
        "web_sources": web_sources,
    })
