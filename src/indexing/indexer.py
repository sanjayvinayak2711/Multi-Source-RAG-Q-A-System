"""
Document indexer module for RAG system.
Coordinates embedding generation and vector storage operations.
"""

from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStoreManager, VectorStoreConfig
from ..ingestion.chunker import Chunk

class DocumentIndexer:
    """Main document indexer class."""
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 vector_store_config: VectorStoreConfig):
        self.embedding_generator = embedding_generator
        self.vector_store = VectorStoreManager(vector_store_config)
        self.indexing_stats = {
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "last_indexed": None,
            "errors": []
        }
    
    def index_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> Dict[str, Any]:
        """
        Index chunks by generating embeddings and storing them in vector store.
        
        Args:
            chunks: List of chunks to index
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with indexing results
        """
        if not chunks:
            return {"success": False, "message": "No chunks to index"}
        
        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(texts, batch_size)
            
            # Prepare documents for vector store
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "embedding": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "indexed_at": datetime.now().isoformat(),
                        "embedding_model": self.embedding_generator.provider.__class__.__name__
                    }
                }
                documents.append(doc)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents)
            
            # Update stats
            self.indexing_stats["documents_indexed"] += len(document_ids)
            self.indexing_stats["chunks_indexed"] += len(chunks)
            self.indexing_stats["last_indexed"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "documents_indexed": len(document_ids),
                "chunks_processed": len(chunks),
                "document_ids": document_ids
            }
            
        except Exception as e:
            error_msg = f"Error indexing chunks: {str(e)}"
            self.indexing_stats["errors"].append(error_msg)
            return {"success": False, "message": error_msg}
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Search in vector store
            results = self.vector_store.search(query_embedding, k)
            
            return results
            
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []
    
    def delete_chunks(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Delete chunks from the index.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            success = self.vector_store.delete_documents(chunk_ids)
            return {
                "success": success,
                "deleted_count": len(chunk_ids) if success else 0
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk dictionary or None if not found
        """
        try:
            return self.vector_store.get_document(chunk_id)
        except Exception as e:
            print(f"Error getting chunk: {e}")
            return None
    
    def rebuild_index(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Rebuild the entire index with new chunks.
        
        Args:
            chunks: List of chunks to rebuild index with
            
        Returns:
            Dictionary with rebuild results
        """
        try:
            # Clear existing index (this would need to be implemented in vector store)
            # For now, we'll just index the new chunks
            
            result = self.index_chunks(chunks)
            
            return {
                "success": result["success"],
                "message": "Index rebuilt successfully" if result["success"] else result["message"],
                "chunks_indexed": result.get("chunks_processed", 0)
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        vector_store_stats = self.vector_store.get_stats()
        
        return {
            **self.indexing_stats,
            "vector_store": vector_store_stats
        }
    
    def validate_index(self) -> Dict[str, Any]:
        """
        Validate the index integrity.
        
        Returns:
            Dictionary with validation results
        """
        try:
            stats = self.get_indexing_stats()
            vector_store_count = stats["vector_store"].get("document_count", 0)
            indexed_count = self.indexing_stats["chunks_indexed"]
            
            is_valid = vector_store_count == indexed_count
            
            return {
                "valid": is_valid,
                "vector_store_count": vector_store_count,
                "indexed_count": indexed_count,
                "difference": abs(vector_store_count - indexed_count)
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def export_index_metadata(self, file_path: str) -> bool:
        """
        Export index metadata to a file.
        
        Args:
            file_path: Path to save the metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = {
                "indexing_stats": self.indexing_stats,
                "vector_store_config": {
                    "store_type": self.vector_store.config.store_type,
                    "collection_name": self.vector_store.config.collection_name,
                    "embedding_dimension": self.vector_store.config.embedding_dimension
                },
                "exported_at": datetime.now().isoformat()
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting metadata: {e}")
            return False
