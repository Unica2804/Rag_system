import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import your custom modules
from src.chunking import split_documents
from src.embedding import EmbeddingProcessor
from src.Ingestion import data_ingestor
from src.retriever import SemanticRetriever
from src.vector_store import VectorStore

load_dotenv()

# Initialize Global Components
llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.3,
    max_retries=2,
    max_tokens=1024    
)

# Initialize Embedding Processor and Vector Store
embedding_processor = EmbeddingProcessor(device='cuda', batch_size=64)
vector_store = VectorStore() 

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG System is Live"}

@app.post("/upload")
async def upload_process(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, saves it temporarily, processes it, 
    and adds it to the persistent Vector Store.
    """
    temp_file_path = f"{file.filename}"
    
    try:
        # Save uploaded file to disk temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file
        doc = data_ingestor(temp_file_path).ingest()
        splitted_doc = split_documents(doc)
        
        # Generate Embeddings
        text_content = [doc.page_content for doc in splitted_doc]
        embeddings = embedding_processor.generate_doc_embeddings(text_content)
        
        # Add to ChromaDB (Persistent)
        vector_store.add_documents(splitted_doc, embeddings)
        
        return JSONResponse(status_code=200, content={'Status': 'Success', 'Chunks Processed': len(splitted_doc)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    finally:
        # Cleanup: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query")
def rag_simple(question: str, top_k: int = 5):
    """
    Query the vector store. 
    """
    try:
        all_docs= vector_store.collection.get()
        if not all_docs['documents']:
            raise HTTPException(status_code=400, detail="No documents in the vector store. Please upload documents first.")
        chunked_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]
        # Initialize retrievers
        semantic_retriever = SemanticRetriever(vector_store=vector_store, embedding_manager=embedding_processor,k=top_k)
        keyword_retriever = BM25Retriever.from_documents(chunked_docs)
        keyword_retriever.k = 2
        retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
        
        # Retrieve
        # results = retriever.invoke(question)
        #
        template = """Use the following context to answer the question.
    
        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Define a helper to format docs
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        # Build the Chain 
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run it
        response = chain.invoke(question)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))