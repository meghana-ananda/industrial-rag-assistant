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

DOC_RELEVANCE_THRESHOLD = 1.0

def get_relevant_docs(question):
    """Search FAISS and return only docs with similarity score below threshold."""
    results = vectorstore.similarity_search_with_score(question, k=5)
    return [doc for doc, score in results if score < DOC_RELEVANCE_THRESHOLD]

def query_rag(question):
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print('='*80)

    context_parts = []

    print("\nSearching documents...")
    docs = get_relevant_docs(question)

    if docs:
        context_parts.append("=== From Documents ===\n" + "\n\n".join([d.page_content for d in docs]))
        print("Found in documents:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
    else:
        print("No relevant docs found — searching the web...")
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
    print("Searches documents first, falls back to web if not found.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        query_rag(question)