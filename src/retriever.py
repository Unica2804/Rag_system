# from vector_store import VectorStore
# from embedding import EmbeddingProcessor
from typing import List,Dict,Any
from pydantic import ConfigDict,Field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class SemanticRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that performs semantic search using embeddings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vector_store: Any = Field(...)
    embedding_manager: Any = Field(...)
    k: int = 5
    threshold: float = 0.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Embeds the query and performs cosine similarity search in the vector store.
        
        Args:
            query: The user query to search for
            run_manager: LangChain callback manager
            
        Returns:
            List of LangChain Documents with metadata
        """
        print(f"USER query: {query}")
        print(f"Top_k: {self.k}, Threshold_score: {self.threshold}")

        # Generate embeddings for query
        query_embedding = self.embedding_manager.generate_query_embeddings(query)

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.k
            )

            langchain_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= self.threshold:
                        doc = Document(
                            page_content=document,
                            metadata={
                                **metadata,
                                'id': doc_id,
                                'similarity_score': similarity_score,
                                'distance': distance,
                                'rank': i + 1
                            }
                        )
                        langchain_docs.append(doc)

                print(f"Retrieved {len(langchain_docs)} documents after filtering")
            else:
                print("No documents found!")

            return langchain_docs

        except Exception as e:
            print(f"Exception occurred: {e}")
            return []