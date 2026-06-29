import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def ingest_documents(docs_folder):
    all_chunks = []
    
    print("Loading all PDFs...")
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, filename)
            print(f"  Loading: {filename}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_chunks.extend(documents)
    
    print(f"Total pages loaded: {len(all_chunks)}")
    
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_chunks)
    print(f"Total chunks created: {len(chunks)}")
    
    print("Creating embeddings and storing...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Done! Vectorstore saved.")

if __name__ == "__main__":
    ingest_documents("industrial_rag")