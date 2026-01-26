import streamlit as st
import requests

# API base URL
API_BASE = "http://localhost:8000"

def upload_file(file):
    """Upload file to /upload endpoint."""
    try:
        files = {'file': (file.name, file, 'application/pdf')}
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def query_rag(question, top_k=5):
    """Query /query endpoint."""
    try:
        data = {'question': question, 'top_k': top_k}
        response = requests.post(f"{API_BASE}/query", json=data, timeout=30)
        response.raise_for_status()
        return response.json().get('response', 'No response')
    except requests.RequestException as e:
        st.error(f"Query failed: {str(e)}")
        return None

# UI Layout
st.title("RAG Query System")
st.markdown("Upload a PDF and ask questions based on its content.")

# Sidebar: Upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=['pdf'])
if st.sidebar.button("Upload Document") and uploaded_file:
    with st.spinner("Uploading and processing..."):
        result = upload_file(uploaded_file)
        if result:
            chunks = result.get('Chunks Processed', 0)
            st.sidebar.success(f"Uploaded! {chunks} chunks processed.")
        else:
            st.sidebar.error("Upload error.")

# Main: Query
st.header("Ask a Question")
question = st.text_input("Enter your question")
top_k = st.slider("Top K results", 1, 10, 5)
if st.button("Submit Query") and question:
    with st.spinner("Retrieving and generating answer..."):
        answer = query_rag(question, top_k)
        if answer:
            st.write("**Answer:**", answer)
        else:
            st.write("No answer received.")