import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

load_dotenv()

def query_rag(question):
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print('='*80)
    
    # Load vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Get relevant documents
    docs = retriever.invoke(question)
    
    # Create context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create LLM
    llm = OllamaLLM(model="mistral")

    # Create prompt
    prompt_text = f"""Answer the question based ONLY on this context:

{context}

Question: {question}

Answer:"""

    # Get answer
    message = llm.invoke(prompt_text)

    print(f"\nAnswer:\n{message}")
    print(f"\nSources:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    print("Industrial RAG Assistant — type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        query_rag(question)