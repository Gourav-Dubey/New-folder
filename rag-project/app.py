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
# from langchain_community.tools.tavily_search import TavilySearchResults

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
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found in environment variables")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in environment variables")
# if not TAVILY_API_KEY:
#     raise ValueError("TAVILY_API_KEY not found in environment variables")

# # ✅ Cloud Embeddings
# embeddings = HuggingFaceEndpointEmbeddings(
#     huggingfacehub_api_token=HF_TOKEN,
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )

# # ✅ Search Tool
# search_tool = TavilySearchResults(max_results=5)

# db_instance = None


# class QuestionRequest(BaseModel):
#     question: str

# class SearchRequest(BaseModel):
#     query: str


# # =========================
# # 📤 UPLOAD ROUTE
# # =========================
# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     global db_instance

#     try:
#         db_instance = None
#         pdf_path = "/tmp/uploaded.pdf"
#         with open(pdf_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=20
#         )
#         chunks = splitter.split_documents(documents)

#         db_instance = Chroma.from_documents(chunks, embeddings)

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

#     llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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


# # =========================
# # 🔍 WEB SEARCH ROUTE
# # =========================
# @app.post("/search")
# async def web_search(request: SearchRequest):
#     try:
#         llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
#         results = search_tool.invoke(request.query)

#         context = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])

#         prompt = ChatPromptTemplate.from_template("""
# You are a helpful assistant. Answer based on these web search results.

# Rules:
# - Respond in English only
# - Be concise and clear
# - Max 6-8 lines

# Search Results:
# {context}

# Question:
# {question}

# Answer:
# """)

#         chain = prompt | llm | StrOutputParser()

#         answer = chain.invoke({"context": context, "question": request.query})
#         return {
#             "answer": answer,
#             "sources": [r['url'] for r in results]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # =========================
# # 🤖 AGENT ROUTE
# # =========================
# from agent import run_agent

# class AgentRequest(BaseModel):
#     message: str

# @app.post("/agent")
# async def agent_chat(request: AgentRequest):
#     try:
#         response = run_agent(request.message)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return {"status": "RAG API running!"}


from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import base64

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from agent import run_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

db_instance = None
current_file_type = None  # "pdf" or "image"
current_image_base64 = None

search_tool = TavilySearchResults(max_results=4)


class ChatRequest(BaseModel):
    message: str


# =========================
# 📤 UPLOAD ROUTE
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global db_instance, current_file_type, current_image_base64

    file_ext = file.filename.lower().split(".")[-1]

    # IMAGE
    if file_ext in ["jpg", "jpeg", "png", "webp", "gif"]:
        contents = await file.read()
        current_image_base64 = base64.b64encode(contents).decode("utf-8")
        current_file_type = "image"
        db_instance = None
        return {"message": f"Image '{file.filename}' uploaded!", "type": "image"}

    # PDF
    elif file_ext == "pdf":
        current_image_base64 = None
        current_file_type = "pdf"
        db_instance = None

        pdf_path = "/tmp/uploaded.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        db_instance = Chroma.from_documents(chunks, embeddings)

        return {"message": f"PDF '{file.filename}' uploaded!", "type": "pdf", "chunks": len(chunks)}

    else:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, PNG supported")


# =========================
# 💬 SMART CHAT ROUTE
# =========================
@app.post("/chat")
async def smart_chat(request: ChatRequest):
    global db_instance, current_file_type, current_image_base64

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    message = request.message.lower()

    # 🖼️ IMAGE mode
    if current_file_type == "image" and current_image_base64:
        llm_vision = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_image_base64}"}},
            {"type": "text", "text": request.message}
        ])
        response = llm_vision.invoke([msg])
        return {"answer": response.content, "mode": "image"}

    # 📄 PDF mode
    if current_file_type == "pdf" and db_instance:
        retriever = db_instance.as_retriever()
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the PDF context.
If not found in PDF but it's a general question, answer from your knowledge.
Be concise and clear.

Context: {context}
Question: {question}
Answer:""")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        answer = chain.invoke(request.message)
        return {"answer": answer, "mode": "pdf"}

    # 🤖 AGENT mode — todos, notes
    agent_keywords = ["todo", "task", "note", "remind", "add", "save", "show", "list", "delete", "complete"]
    if any(kw in message for kw in agent_keywords):
        answer = run_agent(request.message)
        return {"answer": answer, "mode": "agent"}

    # 🔍 WEB SEARCH — default
    try:
        results = search_tool.invoke(request.message)
        context = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on search results.
Be concise, clear, max 6 lines.

Search Results: {context}
Question: {question}
Answer:""")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": request.message})
        sources = [r['url'] for r in results]
        return {"answer": answer, "mode": "web", "sources": sources}
    except:
        # Fallback — LLM direct
        response = llm.invoke(request.message)
        return {"answer": response.content, "mode": "general"}


@app.get("/")
def root():
    return {"status": "Smart AI API running!"}
