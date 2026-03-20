"""
Document preprocessing module for RAG system.
Handles text cleaning, normalization, and preprocessing tasks.
"""

import re
from typing import List, Dict, Any
from .chunker import Chunk

class DocumentPreprocessor:
    """Handles document preprocessing tasks."""
    
    def __init__(self, 
                 clean_whitespace: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 normalize_text: bool = True):
        self.clean_whitespace = clean_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_text = normalize_text
    
    def preprocess_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply preprocessing to a list of chunks.
        
        Args:
            chunks: List of chunks to preprocess
            
        Returns:
            List of preprocessed chunks
        """
        processed_chunks = []
        
        for chunk in chunks:
            processed_content = self.preprocess_text(chunk.content)
            
            processed_chunk = Chunk(
                content=processed_content,
                metadata={
                    **chunk.metadata,
                    "preprocessed": True,
                    "original_length": len(chunk.content),
                    "processed_length": len(processed_content)
                },
                chunk_id=chunk.chunk_id,
                start_index=chunk.start_index,
                end_index=chunk.end_index
            )
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing steps to a single text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return text
        
        processed = text
        
        if self.remove_urls:
            processed = self._remove_urls(processed)
        
        if self.remove_emails:
            processed = self._remove_emails(processed)
        
        if self.clean_whitespace:
            processed = self._clean_whitespace(processed)
        
        if self.normalize_text:
            processed = self._normalize_text(processed)
        
        return processed
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        text = text.strip()
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (basic normalization)."""
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        # Normalize repeated punctuation
        text = re.sub(r'[\.\!\?]{2,}', '.', text)
        return text
    
    def filter_chunks_by_length(self, chunks: List[Chunk], 
                               min_length: int = 50, 
                               max_length: int = 2000) -> List[Chunk]:
        """
        Filter chunks by length criteria.
        
        Args:
            chunks: List of chunks to filter
            min_length: Minimum chunk length
            max_length: Maximum chunk length
            
        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            content_length = len(chunk.content)
            if min_length <= content_length <= max_length:
                filtered_chunk = Chunk(
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "length_filtered": True,
                        "content_length": content_length
                    },
                    chunk_id=chunk.chunk_id,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index
                )
                filtered_chunks.append(filtered_chunk)
        
        return filtered_chunks
    
    def remove_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks based on content similarity.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated list of chunks
        """
        seen_contents = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Simple exact match deduplication
            # In practice, you might want fuzzy matching
            content_hash = hash(chunk.content.lower().strip())
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunk = Chunk(
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "deduplicated": True
                    },
                    chunk_id=chunk.chunk_id,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index
                )
                unique_chunks.append(unique_chunk)
        
        return unique_chunks
