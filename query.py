import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from duckduckgo_search import DDGS

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="mistral")

def web_search(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))

def query_rag(question, mode="both"):
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print('='*80)

    context_parts = []

    if mode in ("docs", "both"):
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
        if docs:
            context_parts.append("=== From Documents ===\n" + "\n\n".join([d.page_content for d in docs]))
            print("\nDocument sources:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")

    if mode in ("web", "both"):
        print("\nSearching the web...")
        results = web_search(question)
        if results:
            web_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
            context_parts.append("=== From Web ===\n" + web_text)
            print("Web sources:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.get('href', '')}")

    context = "\n\n".join(context_parts)
    prompt = f"""Answer the question based on the context below.

{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    print("Industrial RAG Assistant — type 'exit' to quit.")
    print("Search modes: 'docs', 'web', 'both' (default: both)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        mode_input = input("Mode [docs/web/both]: ").strip().lower()
        mode = mode_input if mode_input in ("docs", "web", "both") else "both"
        query_rag(question, mode=mode)