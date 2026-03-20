"""
Document chunking module for RAG system.
Handles splitting documents into manageable chunks for embedding and retrieval.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int

class DocumentChunker:
    """Handles document chunking strategies."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 strategy: str = "recursive"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents with content and metadata
            
        Returns:
            List of chunks
        """
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            chunks = self.chunk_single_document(document, doc_idx)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def chunk_single_document(self, document: Dict[str, Any], doc_idx: int) -> List[Chunk]:
        """Chunk a single document."""
        content = document["content"]
        metadata = document["metadata"]
        
        if self.strategy == "recursive":
            return self._recursive_chunk(content, metadata, doc_idx)
        elif self.strategy == "fixed_size":
            return self._fixed_size_chunk(content, metadata, doc_idx)
        elif self.strategy == "semantic":
            return self._semantic_chunk(content, metadata, doc_idx)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _recursive_chunk(self, content: str, metadata: Dict[str, Any], doc_idx: int) -> List[Chunk]:
        """Recursive character-based chunking with smart splitting."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        
        def _split_text(text: str, separators: List[str]) -> List[str]:
            if len(text) <= self.chunk_size:
                return [text]
            
            for sep in separators:
                if sep in text:
                    parts = text.split(sep)
                    result = []
                    current = ""
                    
                    for part in parts:
                        if len(current + part + sep) <= self.chunk_size:
                            current += part + sep
                        else:
                            if current:
                                result.append(current.rstrip())
                            if len(part) <= self.chunk_size:
                                current = part + sep
                            else:
                                # Part is still too long, split recursively
                                sub_parts = _split_text(part, separators[1:])
                                result.extend(sub_parts[:-1])
                                current = sub_parts[-1] + sep
                    
                    if current:
                        result.append(current.rstrip())
                    return result
            
            # If no separators work, split by character count
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        split_chunks = _split_text(content, separators)
        
        for i, chunk_content in enumerate(split_chunks):
            chunk = Chunk(
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "document_index": doc_idx,
                    "chunk_strategy": self.strategy
                },
                chunk_id=f"{metadata.get('file_name', 'doc')}_{doc_idx}_chunk_{i}",
                start_index=0,  # Could be calculated more precisely
                end_index=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunk(self, content: str, metadata: Dict[str, Any], doc_idx: int) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunk = Chunk(
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "document_index": doc_idx,
                    "chunk_strategy": self.strategy
                },
                chunk_id=f"{metadata.get('file_name', 'doc')}_{doc_idx}_chunk_{len(chunks)}",
                start_index=start,
                end_index=end
            )
            chunks.append(chunk)
            
            if end >= len(content):
                break
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _semantic_chunk(self, content: str, metadata: Dict[str, Any], doc_idx: int) -> List[Chunk]:
        """Semantic chunking based on sentence boundaries."""
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "document_index": doc_idx,
                            "chunk_strategy": self.strategy
                        },
                        chunk_id=f"{metadata.get('file_name', 'doc')}_{doc_idx}_chunk_{len(chunks)}",
                        start_index=0,
                        end_index=len(current_chunk.strip())
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence + " "
        
        if current_chunk:
            chunk = Chunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "document_index": doc_idx,
                    "chunk_strategy": self.strategy
                },
                chunk_id=f"{metadata.get('file_name', 'doc')}_{doc_idx}_chunk_{len(chunks)}",
                start_index=0,
                end_index=len(current_chunk.strip())
            )
            chunks.append(chunk)
        
        return chunks
