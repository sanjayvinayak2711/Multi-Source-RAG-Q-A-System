"""
Vector store management module for RAG system.
Handles vector database operations including storage, retrieval, and similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: str = "chroma"
    collection_name: str = "documents"
    persist_directory: str = "data/vector_store"
    embedding_dimension: int = 1536

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        pass

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except ImportError:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to ChromaDB."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(ids)}")
            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            metadatas.append(doc.get("metadata", {}))
            documents_text.append(doc.get("content", ""))
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )
        
        return ids
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity score
                "distance": results["distances"][0][i]
            }
            documents.append(doc)
        
        return documents
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID from ChromaDB."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            results = self.collection.get(
                ids=[document_id],
                include=["metadatas", "documents"]
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
        except Exception as e:
            print(f"Error getting document: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.documents = {}
        self.embeddings = {}
        self.next_id = 0
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to in-memory store."""
        ids = []
        for doc in documents:
            doc_id = doc.get("id", f"doc_{self.next_id}")
            if doc_id not in self.documents:
                self.next_id += 1
            
            self.documents[doc_id] = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "embedding": doc["embedding"]
            }
            ids.append(doc_id)
        
        return ids
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        if not self.documents:
            return []
        
        similarities = []
        query_vec = np.array(query_embedding)
        
        for doc_id, doc in self.documents.items():
            doc_vec = np.array(doc["embedding"])
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((doc_id, similarity, doc))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, similarity, doc in similarities[:k]:
            results.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(similarity),
                "distance": 1 - float(similarity)
            })
        
        return results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from in-memory store."""
        for doc_id in document_ids:
            self.documents.pop(doc_id, None)
        return True
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        return self.documents.get(document_id)

class VectorStoreManager:
    """Manager class for vector store operations."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self) -> VectorStore:
        """Create vector store based on configuration."""
        if self.config.store_type.lower() == "chroma":
            return ChromaVectorStore(self.config)
        elif self.config.store_type.lower() == "memory":
            return InMemoryVectorStore(self.config)
        else:
            raise ValueError(f"Unknown vector store type: {self.config.store_type}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        return self.vector_store.add_documents(documents)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.vector_store.similarity_search(query_embedding, k)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        return self.vector_store.delete_documents(document_ids)
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self.vector_store.get_document_by_id(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if hasattr(self.vector_store, 'get_collection_stats'):
            return self.vector_store.get_collection_stats()
        return {"document_count": len(self.vector_store.documents) if hasattr(self.vector_store, 'documents') else 0}
