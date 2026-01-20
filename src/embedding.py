# from sentence_transformers import SentenceTransformer
import numpy
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Tuple
_model_cache={}

class EmbeddingProcessor:
    def __init__(self,model:str='BAAI/bge-small-en-v1.5',device:str='cpu',batch_size:int=16):
        self.model_name=model
        self.device=device
        self.batch_size=batch_size
        self.model=None
        self._load_model()
        
    def _load_model(self):
        global _model_cache
        
        # Return cached model if already loaded
        if self.model_name in _model_cache:
            print(f"Using cached model {self.model_name}")
            self.model = _model_cache[self.model_name]
            return
        try:
            print(f"loading model {self.model_name}")
            self.model=HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device}, 
                encode_kwargs={"normalize_embeddings": True, "batch_size": self.batch_size}
                )
            _model_cache[self.model_name]=self.model
            print(f"Loaded model {self.model_name} with dimensions : {self.model._client.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"{self.model_name} error : {e}")
            raise
            
    def generate_doc_embeddings(self,texts:List[str])-> numpy.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating Embeddings for {len(texts)} texts....")
        embeddings=self.model.embed_documents(texts)
        print(f"Generated embedding with shape: {len(embeddings)}")
        return numpy.array(embeddings)
    
    def generate_query_embeddings(self,texts:str)-> numpy.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating Embeddings for query....")
        embeddings=self.model.embed_query(texts)
        print(f"Generated embedding with shape: {len(embeddings)}")
        return numpy.array(embeddings)
    def get_device(self) -> str:
        """Return the device the model is running on."""
        return str(self.model._client.device)