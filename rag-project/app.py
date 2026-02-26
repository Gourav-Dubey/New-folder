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

# # ✅ CORS
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

# # ✅ Check Environment Variables
# HF_TOKEN = os.getenv("HF_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found in environment variables")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in environment variables")

# # ✅ Cloud Embeddings (No local torch needed)
# embeddings = HuggingFaceEndpointEmbeddings(
#     huggingfacehub_api_token=HF_TOKEN,
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_instance = None


# class QuestionRequest(BaseModel):
#     question: str


# # =========================
# # 📤 UPLOAD ROUTE
# # =========================
# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     global db_instance

#     try:
#         # Reset memory
#         db_instance = None

       
        

#         # Save new PDF
#         pdf_path = "/tmp/uploaded.pdf"
#         with open(pdf_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         # Load PDF
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()

#         # Split into chunks
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=20
#         )
#         chunks = splitter.split_documents(documents)

#         # Create NEW DB only for this PDF
#         db_instance = Chroma.from_documents(
#             chunks,
#             embeddings,
            
#         )

#         return {
#             "message": f"PDF uploaded successfully!",
#             "chunks": len(chunks)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # =========================
# # 💬 ASK ROUTE
# # =========================
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     global db_instance

#     if db_instance is None:
#         return {"answer": "Please upload a PDF first."}

#     retriever = db_instance.as_retriever()

#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0
#     )

#     prompt = ChatPromptTemplate.from_template("""
# You are a helpful PDF assistant.

# - Answer ONLY from the context.
# - If answer not found, say:
#   "This information is not present in the uploaded PDF."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     answer = chain.invoke(request.question)
#     return {"answer": answer}


    

# @app.get("/")
# def root():
#     return {"status": "RAG API running!"} 


# uvicorn app:app --reload

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
from langchain_community.tools.tavily_search import TavilySearchResults

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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# ✅ Cloud Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Search Tool
search_tool = TavilySearchResults(max_results=5)

db_instance = None


class QuestionRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str


# =========================
# 📤 UPLOAD ROUTE
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global db_instance

    try:
        db_instance = None
        pdf_path = "/tmp/uploaded.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20
        )
        chunks = splitter.split_documents(documents)

        db_instance = Chroma.from_documents(chunks, embeddings)

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

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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


# =========================
# 🔍 WEB SEARCH ROUTE
# =========================
@app.post("/search")
async def web_search(request: SearchRequest):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        results = search_tool.invoke(request.query)

        context = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])

        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on these web search results.

Rules:
- Respond in English only
- Be concise and clear
- Max 6-8 lines

Search Results:
{context}

Question:
{question}

Answer:
""")

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": request.query})
        return {
            "answer": answer,
            "sources": [r['url'] for r in results]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "RAG API running!"}
