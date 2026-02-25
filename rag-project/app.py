# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import shutil
# import os

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# # ✅ Proper CORS Setup
# origins = [
#     "http://localhost:5173",
#     "https://pdfchatbo.netlify.app", 
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Check HF Token
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found in environment variables")

# # ✅ Embedding (Cloud Based - No Local Model)
# embeddings = HuggingFaceEndpointEmbeddings(
#     huggingfacehub_api_token=HF_TOKEN,
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_instance = None

# class QuestionRequest(BaseModel):
#     question: str


# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     global db_instance

#     try:
#         # Save uploaded file
#         with open("uploaded.pdf", "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         db_instance = None

#         # Delete old DB
#         if os.path.exists("./chroma_db"):
#             shutil.rmtree("./chroma_db", ignore_errors=True)

#         # Load PDF
#         loader = PyPDFLoader("uploaded.pdf")
#         documents = loader.load()

#         # Split into chunks
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=20
#         )
#         chunks = splitter.split_documents(documents)

#         # Create new vector DB
#         db_instance = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="/tmp/chroma_db"
#         )

#         return {"message": f"PDF uploaded! {len(chunks)} chunks processed."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     global db_instance

#     if db_instance is None:
#         db_instance = Chroma(
#             persist_directory="/tmp/chroma_db",
#             embedding_function=embeddings
#         )

#     retriever = db_instance.as_retriever()

#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0
#     )

#     prompt = ChatPromptTemplate.from_template("""
# You are a helpful PDF assistant. Answer based ONLY on the context provided.

# Rules:
# - ALWAYS respond in English only
# - Be concise and to the point
# - If asked casual questions like "hello", just greet back simply
# - Use bullet points only when listing multiple items
# - Max 8-10 lines for simple questions
# - Don't add unnecessary filler text

# Context: {context}

# Question: {question}

# Answer:
# """)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     try:
#         answer = chain.invoke(request.question)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/")
# def root():
#     return {"status": "RAG API running!"} 

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ✅ CORS
origins = [
    "http://localhost:5173",
    "https://pdfchatbo.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Check Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# ✅ Cloud Embeddings (No local torch needed)
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

db_instance = None


class QuestionRequest(BaseModel):
    question: str


# =========================
# 📤 UPLOAD ROUTE
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global db_instance

    try:
        # Reset memory
        db_instance = None

       
        

        # Save new PDF
        pdf_path = "/tmp/uploaded.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20
        )
        chunks = splitter.split_documents(documents)

        # Create NEW DB only for this PDF
        db_instance = Chroma.from_documents(
            chunks,
            embeddings,
            
        )

        return {
            "message": f"PDF uploaded successfully!",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 💬 ASK ROUTE
# =========================
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global db_instance

    if db_instance is None:
        return {"answer": "Please upload a PDF first."}

    retriever = db_instance.as_retriever()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant.

- Answer ONLY from the context.
- If answer not found, say:
  "This information is not present in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
""")

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