# RAG_system — Retrieval-Augmented Generation System

**Status:** ⚠️ **Under development**

> RAG_system is a small, modular project that demonstrates a Retrieval-Augmented Generation (RAG) pipeline. It provides utilities for ingesting documents, chunking and embedding text, storing embeddings in a vector store, and running retrieval + generation flows with local LLM and API.

---

## Quick Start

Prerequisites:
- Python 3.11+
- Git
- (Optional) Docker

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

Build the Docker image (optional):

```bash
docker build -t <Name of image> .
```

Run the Docker image (optional):

```bash
docker run -p 8000:8000 <Name of image>
```

In browser open http://127.0.0.1:8000/docs to open swagger UI.

Run the main pipeline (example):

```bash
uvicorn main:app --port 8000
```

Create a env file:

```bash
GROQ_API_KEY=
```

---

##  Project Structure

- `main.py` — entry point demonstrating end-to-end usage
- `src/` — core modules:
  - `chunking.py` — document chunking utilities
  - `embedding.py` — embedding helpers
  - `Ingestion.py` — document ingestion routines
  - `retriever.py` — retrieval logic
  - `vector_store.py` — vector store adapter (Chroma)
- `unit_tests/` — Contains tests for checking functionality of some modules
- `data/` — persistent vector store
- `requirements.txt` — Python dependencies
- `Dockerfile` — containerization configuration

---

##  How it works (high level)

1. Ingest documents from a temp folder (PDFs, text, docs)
2. Chunk documents into smaller passages for embedding
3. Create embeddings using the configured encoder
4. Store embeddings in the vector store
5. Query the vector store to retrieve relevant context
6. Combine retrieved context with the user's prompt and call the LLM

---

##  Development & Roadmap

The project is actively being developed. Planned improvements:
- Add tests and CI for core components 
- Improve ingestion for more file formats (HTML)
- Add configuration for multiple embedding/LLM providers
- Robust Docker and deployment workflows
- Example notebooks and end-to-end demos