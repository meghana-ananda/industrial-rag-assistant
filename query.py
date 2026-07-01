import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from ddgs import DDGS
from hybrid_retriever import HybridRetriever

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = HybridRetriever(vectorstore, k=5)
llm = OllamaLLM(model="mistral")

DOC_RELEVANCE_THRESHOLD = 1.0

def web_search(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))

def get_relevant_docs(question):
    return retriever.retrieve(question, threshold=DOC_RELEVANCE_THRESHOLD)

def query_rag(question):
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print('='*80)

    print("\nSearching documents...")
    docs = get_relevant_docs(question)
    context = ""

    if docs:
        context = "\n\n".join([d.page_content for d in docs])
        print("Found in documents:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
    else:
        print("No relevant docs found — searching the web...")
        results = web_search(question)
        if results:
            context = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
            print("Web sources:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.get('href', '')}")

    if not context:
        print("\nAnswer:\nI couldn't find relevant information in the documents or on the web for this question.")
        return

    prompt = f"""Answer the question based ONLY on the context below. Do not use your own knowledge.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
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
