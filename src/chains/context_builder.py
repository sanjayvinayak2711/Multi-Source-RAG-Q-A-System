"""
Context builder module for RAG system.
Handles context assembly and optimization for LLM input.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_context_length: int = 4000
    max_documents: int = 5
    include_metadata: bool = True
    include_scores: bool = True
    context_format: str = "document"  # "document", "paragraph", "sentence"
    deduplicate: bool = True

class ContextBuilder:
    """Builds and optimizes context for LLM input."""
    
    def __init__(self, config: ContextConfig = None):
        self.config = config or ContextConfig()
    
    def build_context(self, 
                     retrieved_docs: List[Dict[str, Any]], 
                     question: str = None) -> Dict[str, Any]:
        """
        Build optimized context from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            question: Optional question for context optimization
            
        Returns:
            Dictionary with built context and metadata
        """
        if not retrieved_docs:
            return {
                "context": "",
                "metadata": {
                    "document_count": 0,
                    "context_length": 0,
                    "truncated": False
                }
            }
        
        # Step 1: Preprocess documents
        processed_docs = self._preprocess_documents(retrieved_docs)
        
        # Step 2: Sort by relevance
        sorted_docs = self._sort_by_relevance(processed_docs)
        
        # Step 3: Deduplicate if enabled
        if self.config.deduplicate:
            sorted_docs = self._deduplicate_documents(sorted_docs)
        
        # Step 4: Format context
        context, metadata = self._format_context(sorted_docs, question)
        
        return {
            "context": context,
            "metadata": metadata
        }
    
    def _preprocess_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess documents for context building."""
        processed = []
        
        for doc in docs:
            processed_doc = {
                "content": doc.get("content", "").strip(),
                "score": doc.get("score", 0.0),
                "metadata": doc.get("metadata", {}),
                "id": doc.get("id", "")
            }
            
            # Clean content
            processed_doc["content"] = self._clean_content(processed_doc["content"])
            
            # Calculate content length
            processed_doc["length"] = len(processed_doc["content"])
            
            processed.append(processed_doc)
        
        return processed
    
    def _clean_content(self, content: str) -> str:
        """Clean document content."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might confuse the LLM
        content = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}"\'\/\n]', '', content)
        
        # Ensure proper spacing around punctuation
        content = re.sub(r'([.!?])\s*', r'\1 ', content)
        
        return content.strip()
    
    def _sort_by_relevance(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort documents by relevance score."""
        return sorted(docs, key=lambda x: x["score"], reverse=True)
    
    def _deduplicate_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents."""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            # Create a hash of the content (lowercase, stripped)
            content_hash = hash(doc["content"].lower().strip())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _format_context(self, docs: List[Dict[str, Any]], question: str = None) -> tuple[str, Dict[str, Any]]:
        """Format documents into context string."""
        context_parts = []
        current_length = 0
        documents_used = 0
        truncated = False
        
        for i, doc in enumerate(docs):
            if documents_used >= self.config.max_documents:
                break
            
            # Format document based on configuration
            formatted_doc = self._format_single_document(doc, i + 1)
            
            # Check if adding this document would exceed max length
            potential_length = current_length + len(formatted_doc)
            
            if potential_length <= self.config.max_context_length:
                context_parts.append(formatted_doc)
                current_length = potential_length
                documents_used += 1
            else:
                # Try to add a truncated version
                truncated_doc = self._truncate_document(doc, self.config.max_context_length - current_length)
                if truncated_doc:
                    context_parts.append(truncated_doc)
                    current_length += len(truncated_doc)
                    documents_used += 1
                truncated = True
                break
        
        context = "\n\n".join(context_parts)
        
        metadata = {
            "document_count": documents_used,
            "context_length": len(context),
            "max_length": self.config.max_context_length,
            "truncated": truncated,
            "format": self.config.context_format
        }
        
        return context, metadata
    
    def _format_single_document(self, doc: Dict[str, Any], doc_num: int) -> str:
        """Format a single document based on configuration."""
        content = doc["content"]
        
        if self.config.context_format == "document":
            parts = [f"[Document {doc_num}]"]
            
            if self.config.include_scores:
                parts.append(f"[Relevance: {doc['score']:.3f}]")
            
            if self.config.include_metadata and doc["metadata"]:
                metadata_str = self._format_metadata(doc["metadata"])
                if metadata_str:
                    parts.append(f"[Metadata: {metadata_str}]")
            
            parts.append(content)
            return "\n".join(parts)
        
        elif self.config.context_format == "paragraph":
            return f"Document {doc_num}: {content}"
        
        elif self.config.context_format == "sentence":
            # Split into sentences and add document prefix
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            return "\n".join([f"Doc{doc_num}: {sentence}." for sentence in sentences])
        
        else:
            return content
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display."""
        if not metadata:
            return ""
        
        # Select important metadata fields
        important_fields = ["source", "file_name", "title", "author", "date"]
        formatted_parts = []
        
        for field in important_fields:
            if field in metadata and metadata[field]:
                formatted_parts.append(f"{field}={metadata[field]}")
        
        return ", ".join(formatted_parts)
    
    def _truncate_document(self, doc: Dict[str, Any], max_length: int) -> Optional[str]:
        """Truncate document to fit within max length."""
        if max_length <= 100:  # Too short to include anything meaningful
            return None
        
        content = doc["content"]
        
        # Try to truncate at sentence boundaries
        sentences = re.split(r'[.!?]+', content)
        truncated_content = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_length = len(truncated_content) + len(sentence) + 2  # +2 for punctuation and space
            
            if potential_length <= max_length:
                truncated_content += sentence + ". "
            else:
                break
        
        if truncated_content:
            return f"[Document truncated] {truncated_content.strip()}"
        else:
            # If even one sentence is too long, truncate at word boundary
            words = content.split()
            truncated_words = []
            current_length = 0
            
            for word in words:
                potential_length = current_length + len(word) + 1
                if potential_length <= max_length:
                    truncated_words.append(word)
                    current_length = potential_length
                else:
                    break
            
            if truncated_words:
                return f"[Document truncated] {' '.join(truncated_words)}..."
            else:
                return None
    
    def optimize_context_for_question(self, context: str, question: str) -> str:
        """
        Optimize context for a specific question.
        
        Args:
            context: Current context
            question: Target question
            
        Returns:
            Optimized context
        """
        # Extract keywords from question
        question_keywords = self._extract_keywords(question)
        
        # Split context into chunks
        context_chunks = context.split("\n\n")
        
        # Score chunks based on keyword relevance
        scored_chunks = []
        for chunk in context_chunks:
            score = self._calculate_chunk_relevance(chunk, question_keywords)
            scored_chunks.append((chunk, score))
        
        # Sort by relevance and rebuild context
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        optimized_chunks = []
        current_length = 0
        
        for chunk, score in scored_chunks:
            if current_length + len(chunk) <= self.config.max_context_length:
                optimized_chunks.append(chunk)
                current_length += len(chunk)
            else:
                break
        
        return "\n\n".join(optimized_chunks)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return list(set(keywords))
    
    def _calculate_chunk_relevance(self, chunk: str, keywords: List[str]) -> float:
        """Calculate relevance score of a chunk based on keywords."""
        chunk_lower = chunk.lower()
        score = 0
        
        for keyword in keywords:
            # Count occurrences of keyword in chunk
            occurrences = chunk_lower.count(keyword)
            score += occurrences * len(keyword)  # Weight by keyword length
        
        # Normalize by chunk length
        if len(chunk) > 0:
            score = score / len(chunk)
        
        return score
