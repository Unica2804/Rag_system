from sentence_transformers import SentenceTransformer
import numpy
from typing import List, Dict, Any, Tuple

class EmbeddingProcessor:
    def __init__(self,model:str='all-MiniLM-L6-v2'):
        self.model_name=model
        self.model=None
        self._load_model()
        
    def _load_model(self):
        try:
            print(f"loading model {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Loaded model {self.model_name} with dimensions : {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"{self.model_name} error : {e}")
            raise
            
    def generate_embeddings(self,texts:List[str])-> numpy.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating Embeddings for {len(texts)} texts....")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embedding with shape: {embeddings.shape}")
        return embeddings