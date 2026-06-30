import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭")
st.title("🏭 Industrial RAG Assistant")
st.caption("Ask questions about your industrial documents.")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""Answer the question based ONLY on this context:

{context}

Question: {question}

Answer:"""

            llm = OllamaLLM(model="mistral")
            answer = llm.invoke(prompt)

        st.markdown(answer)

        sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
        with st.expander("Sources"):
            for src in sources:
                st.write(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
