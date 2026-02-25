from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def ask_question(question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever()
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below:
    Context: {context}
    Question: {question}
    """)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (q to quit): ")
        if q == "q":
            break
        print("Answer:", ask_question(q))
