from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def ingest_pdf(pdf_path):
    # PDF load karo
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Chunks banao
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Embeddings aur DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    
    print(f"Done! {len(chunks)} chunks saved.")

if __name__ == "__main__":
    ingest_pdf("document.pdf")