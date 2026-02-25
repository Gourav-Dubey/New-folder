from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_instance = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global db_instance
    
    # PDF save karo
    with open("uploaded.pdf", "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Pehle db close karo
    db_instance = None
    
    # Chroma_db folder delete karo
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db", ignore_errors=True)
    
    # Process karo
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Naya DB banao
    db_instance = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    
    return {"message": f"PDF uploaded! {len(chunks)} chunks processed."}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global db_instance
    
    if db_instance is None:
        db_instance = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    retriever = db_instance.as_retriever()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # prompt = ChatPromptTemplate.from_template("""
    # Answer the question based on the context below. Be helpful and detailed.
    # Context: {context}
    # Question: {question}
    # """)
    prompt = ChatPromptTemplate.from_template("""You are a helpful PDF assistant. Answer based ONLY on the context provided.

Rules:
- ALWAYS respond in English only
- Be concise and to the point
- If asked casual questions like "hello", just greet back simply
- Use bullet points only when listing multiple items
- Max 5-6 lines for simple questions
- Don't add unnecessary filler text

Context: {context}

Question: {question}

Answer:""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(request.question)
    return {"answer": answer}

@app.get("/")
def root():
    return {"status": "RAG API running!"}