"""
Transformer-based text embedding models for feature extraction.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceBERTEmbedder:
    """
    Sentence-BERT embeddings using pre-trained models.
    Fast and effective for sentence-level representations.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Sentence-BERT model.
        
        Args:
            model_name: Pre-trained model name
        """
        self.model_name = model_name
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim


class DistilBERTEmbedder:
    """
    DistilBERT embeddings using mean pooling of last hidden states.
    """
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', max_length: int = 512):
        """Initialize DistilBERT model."""
        self.model_name = model_name
        self.max_length = max_length
        
        logger.info(f"Loading DistilBERT model: {model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.dim
        logger.info(f"Model loaded on {self.device}. Embedding dimension: {self.embedding_dim}")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling of token embeddings weighted by attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """Encode texts into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim


class HybridEmbedder:
    """Combines multiple embedding approaches."""
    
    def __init__(self, embedders: List):
        """Initialize hybrid embedder."""
        self.embedders = embedders
        self.embedding_dim = sum(emb.get_embedding_dim() for emb in embedders)
        logger.info(f"Hybrid embedder initialized with {len(embedders)} models")
        logger.info(f"Total embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts using all embedders and concatenate."""
        all_embeddings = []
        
        for i, embedder in enumerate(self.embedders):
            logger.info(f"Encoding with embedder {i+1}/{len(self.embedders)}")
            embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=False)
            all_embeddings.append(embeddings)
        
        return np.hstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of concatenated embeddings."""
        return self.embedding_dim