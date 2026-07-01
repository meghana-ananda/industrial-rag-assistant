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

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭")
st.title("🏭 Industrial RAG Assistant")
st.caption("Searches your industrial documents first, falls back to the web if needed.")

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return HybridRetriever(vs, k=5)

retriever = load_retriever()

def get_relevant_docs(question, threshold=1.0):
    """Hybrid BM25 + semantic retrieval fused with Reciprocal Rank Fusion."""
    return retriever.retrieve(question, threshold=threshold)

def web_search(query, max_results=5):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.sidebar.markdown("### Search Settings")
relevance_threshold = st.sidebar.slider("Doc relevance threshold", 0.5, 2.0, 1.0, 0.1,
    help="Lower = stricter (falls back to web more often). Higher = uses docs even if loosely related.")
use_rewriter = st.sidebar.toggle("Query rewriting", value=True,
    help="Rewrites your question into a keyword-rich query before searching. Improves recall for vague questions.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Auto:** Documents → Web fallback")
st.sidebar.markdown("Searches your PDFs first. Falls back to the web if no relevant content is found.")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            search_query = rewrite_query(question) if use_rewriter else question
            if use_rewriter and search_query != question:
                st.caption(f"Rewritten query: _{search_query}_")
            docs = get_relevant_docs(search_query, threshold=relevance_threshold)
            context_parts = []
            doc_sources = []
            web_sources = []
            source_used = None

            if docs:
                context_parts.append("=== From Documents ===\n" + "\n\n".join([d.page_content for d in docs]))
                doc_sources = list({d.metadata.get("source", "Unknown") for d in docs})
                source_used = "documents"
            else:
                st.toast("No relevant docs found — searching the web...", icon="🌐")
                web_results = web_search(question)
                if web_results:
                    web_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in web_results])
                    context_parts.append("=== From Web ===\n" + web_text)
                    web_sources = [r.get("href", "") for r in web_results if r.get("href")]
                    source_used = "web"

            context = "\n\n".join(context_parts)
            # Include last 4 messages (2 exchanges) so follow-up questions work
            recent_history = st.session_state.messages[-4:] if len(st.session_state.messages) > 1 else []
            history_text = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history]
            ) if recent_history else ""

            history_section = f"\nConversation so far:\n{history_text}\n" if history_text else ""

            prompt = f"""Answer the question based on the context below.{history_section}
Context:
{context}

Question: {question}

Answer:"""

            if not context:
                answer = "I couldn't find relevant information in the documents or on the web for this question."
            else:
                llm = OllamaLLM(model="mistral")
                answer = llm.invoke(prompt)

        st.markdown(answer)

        if source_used == "documents":
            st.caption("📄 Answered from your documents")
        elif source_used == "web":
            st.caption("🌐 Answered from web search (not found in documents)")

        if doc_sources or web_sources:
            with st.expander("Sources"):
                if doc_sources:
                    st.markdown("**Documents:**")
                    for src in doc_sources:
                        st.write(f"- {src}")
                if web_sources:
                    st.markdown("**Web:**")
                    for url in web_sources:
                        st.write(f"- {url}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
