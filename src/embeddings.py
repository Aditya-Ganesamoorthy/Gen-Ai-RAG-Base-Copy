from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch
import warnings
warnings.filterwarnings("ignore")

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize embedding model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def generate_batch_embeddings(self, texts: List[str], batch_size=32) -> np.ndarray:
        """Generate embeddings in batches to handle large texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)