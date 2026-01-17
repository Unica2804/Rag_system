import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import numpy

## Vector Store
class VectorStore:
    """Manages document embeddings in Chroma DB as a Vector Store"""
    def __init__(self, collection_name: str="pdf_documents", persist_directory: str= "data/vector_store"):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"PDF document embedding for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector : {e}")
            raise

    def _generate_doc_id(self, doc) -> str:
        """Generate a deterministic ID based on document content and metadata."""
        # Create a unique string from source file, page number, and content
        source = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 0)
        # Use first 500 chars of content to create hash (enough to be unique)
        content_sample = doc.page_content[:500]
        unique_string = f"{source}_{page}_{content_sample}"
        # Create a hash of the unique string
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _get_existing_ids(self) -> set:
        """Get all existing document IDs in the collection."""
        if self.collection.count() == 0:
            return set()
        # Get all existing IDs
        existing = self.collection.get(include=[])
        return set(existing['ids'])

    def add_documents(self, documents: List[Any], embeddings: numpy.ndarray):

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Processing {len(documents)} documents...")

        # Get existing document IDs to avoid duplicates
        existing_ids = self._get_existing_ids()
        print(f"Found {len(existing_ids)} existing documents in collection")

        ids=[]
        metadatas= []
        documents_text=[]
        embeddings_list=[]
        skipped_count = 0

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate deterministic ID based on content
            doc_id = self._generate_doc_id(doc)

            # Skip if document already exists
            if doc_id in existing_ids:
                skipped_count += 1
                continue

            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        if skipped_count > 0:
            print(f"Skipped {skipped_count} duplicate documents")

        if not ids:
            print("No new documents to add - all documents already exist in the collection")
            print(f"Total documents in collection: {self.collection.count()}")
            return

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(ids)} new documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise