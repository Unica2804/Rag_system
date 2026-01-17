# from vector_store import VectorStore
# from embedding import EmbeddingProcessor
from typing import List,Dict,Any

class RagRetrieval:
    def __init__(self,vector_store, embedding_manager):
        self.vector_store= vector_store
        self.embedding_manager= embedding_manager
    def retrieve(self, query: str, top_k: int=5, threshold_score: float=0.0) -> List[Dict[str,Any]]:
        """
        It Takes a query embeds it and does cosine similarity to search from Vector store to retrieve Content.

        Args:
            query: It takes in the user query
            top_k: It returns the number of best matches
            threshold_score: The score you set for cosine similarity 
        Returns:
            List of retrieved Docs and the metadata

        """
        print(f"USER query: {query}")
        print(f"Top_k: {top_k}, Threshold_score: {threshold_score}")

        # Generate embeddings for query
        query_embedding= self.embedding_manager.generate_embeddings([query])[0]
        # Search in Vector Store
        try:
            results= self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k                
            )

            # Retrieve results
            retrieved_docs=[]

            if results['documents'] and results['documents'][0]:
                documents= results['documents'][0]
                metadatas= results['metadatas'][0]
                distances= results['distances'][0]
                ids= results['ids'][0]

                for i , (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    # Convert distance to simiarity score. Chroma uses cosine distance

                    similarity_score= 1- distance

                    # Condition for doc retrieval

                    if similarity_score >= threshold_score:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i+1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No documents found !!")

            return retrieved_docs

        except Exception as e:
            print(f"Exception occured: {e}")
            return[]

