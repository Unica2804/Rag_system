import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Import your custom modules
from src.chunking import split_documents
from src.embedding import EmbeddingProcessor
from src.Ingestion import data_ingestor
from src.retriever import RagRetrieval
from src.vector_store import VectorStore

load_dotenv()

# 1. Initialize Global Components
# These are loaded once when the app starts.
llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.3,
    max_retries=2,
    max_tokens=1024    
)

embedding_processor = EmbeddingProcessor(device='cuda', batch_size=64)
# Assuming VectorStore() handles the connection to the persistent ChromaDB folder internally
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
        # 1. Save uploaded file to disk temporarily
        # Many PDF parsers need a physical file path, not just bytes in memory.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Process the file
        doc = data_ingestor(temp_file_path).ingest()
        splitted_doc = split_documents(doc)
        
        # 3. Generate Embeddings
        text_content = [doc.page_content for doc in splitted_doc]
        embeddings = embedding_processor.generate_doc_embeddings(text_content)
        
        # 4. Add to ChromaDB (Persistent)
        vector_store.add_documents(splitted_doc, embeddings)
        
        return JSONResponse(status_code=200, content={'Status': 'Success', 'Chunks Processed': len(splitted_doc)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    finally:
        # 5. Cleanup: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query")
def rag_simple(question: str, top_k: int = 5):
    """
    Query the vector store. 
    """
    try:
        # Initialize retriever dynamically using the global vector_store
        retriever = RagRetrieval(vector_store, embedding_processor)
        
        # 1. Retrieve
        results = retriever.retrieve(question, top_k=top_k)
        
        # 2. Construct Context
        # Ensure your retrieval result keys match here (e.g. doc['content'] or doc.page_content)
        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        
        if not context:
            return {"answer": "No relevant context found to answer the question."}

        # 3. Construct Prompt
        # IMPORTANT: Use a standard string (not f-string) for the template
        prompt_template = """Use the following context to answer the user's question.
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        # 4. Invoke LLM
        formatted_prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        
        return {"response": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))