"""
Embedding generation module for RAG system.
Supports various embedding models and providers.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self._dimension = 1536 if model_name == "text-embedding-ada-002" else 1536
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        # Placeholder implementation
        # In real implementation, use openai.Embedding.create()
        embeddings = []
        for text in texts:
            # Mock embedding
            embedding = np.random.rand(self._dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]
    
    def get_embedding_dimension(self) -> int:
        return self._dimension

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Hugging Face embedding provider."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dimension = 384  # Default for MiniLM
        # In real implementation, load the model here
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face models."""
        # Placeholder implementation
        # In real implementation, use transformers library
        embeddings = []
        for text in texts:
            # Mock embedding
            embedding = np.random.rand(self._dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]
    
    def get_embedding_dimension(self) -> int:
        return self._dimension

class EmbeddingGenerator:
    """Main embedding generator class."""
    
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.provider.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        return self.provider.embed_query(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.provider.get_embedding_dimension()

def create_embedding_generator(provider_type: str, **kwargs) -> EmbeddingGenerator:
    """
    Factory function to create embedding generators.
    
    Args:
        provider_type: Type of provider ('openai', 'huggingface')
        **kwargs: Additional arguments for the provider
        
    Returns:
        EmbeddingGenerator instance
    """
    if provider_type.lower() == "openai":
        provider = OpenAIEmbeddingProvider(**kwargs)
    elif provider_type.lower() == "huggingface":
        provider = HuggingFaceEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return EmbeddingGenerator(provider)
