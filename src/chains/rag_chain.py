"""
RAG chain implementation for question answering.
Coordinates retrieval, context building, and answer generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class RAGConfig:
    """Configuration for RAG chain."""
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    include_sources: bool = True
    include_scores: bool = True

class RAGChain:
    """Main RAG chain for question answering."""
    
    def __init__(self, 
                 indexer,
                 llm_client,
                 config: RAGConfig = None):
        self.indexer = indexer
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
    
    def answer_question(self, 
                       question: str, 
                       context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: The question to answer
            context_filter: Optional filter for context retrieval
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_context(question, context_filter)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieved_count": 0
                }
            
            # Step 2: Build context
            context = self._build_context(retrieved_docs)
            
            # Step 3: Generate answer
            answer = self._generate_answer(question, context)
            
            # Step 4: Prepare response
            response = {
                "answer": answer,
                "sources": self._prepare_sources(retrieved_docs) if self.config.include_sources else [],
                "confidence": self._calculate_confidence(retrieved_docs),
                "retrieved_count": len(retrieved_docs),
                "context_length": len(context)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in RAG chain: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _retrieve_context(self, question: str, context_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the question."""
        # Search for similar chunks
        results = self.indexer.search_similar_chunks(question, k=self.config.retrieval_k)
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc in results 
            if doc.get("score", 0) >= self.config.similarity_threshold
        ]
        
        # Apply additional filters if provided
        if context_filter:
            filtered_results = self._apply_filters(filtered_results, context_filter)
        
        return filtered_results
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_content = doc.get("content", "").strip()
            
            # Add document identifier
            doc_prefix = f"[Document {i+1}]"
            
            # Check if adding this document would exceed max length
            potential_length = current_length + len(doc_prefix) + len(doc_content) + 2
            
            if potential_length <= self.config.max_context_length:
                context_parts.append(f"{doc_prefix} {doc_content}")
                current_length = potential_length
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        # This is a placeholder implementation
        # In real implementation, you would use the LLM client
        
        prompt = f"""Based on the following context, please answer the question. 
If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {question}

Answer:"""

        # Mock LLM response
        # In real implementation: response = self.llm_client.generate(prompt)
        
        if "test" in question.lower():
            return "This is a test answer based on the provided context."
        else:
            return f"Based on the context provided, here's an answer to your question about: {question[:50]}..."
    
    def _prepare_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for response."""
        sources = []
        
        for doc in retrieved_docs:
            source_info = {
                "id": doc.get("id", ""),
                "score": doc.get("score", 0),
                "metadata": doc.get("metadata", {})
            }
            
            if self.config.include_scores:
                source_info["similarity_score"] = doc.get("score", 0)
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not retrieved_docs:
            return 0.0
        
        # Simple confidence calculation based on average similarity score
        scores = [doc.get("score", 0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)
        
        # Adjust confidence based on number of relevant documents
        doc_count_factor = min(len(retrieved_docs) / self.config.retrieval_k, 1.0)
        
        return avg_score * doc_count_factor
    
    def _apply_filters(self, docs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply additional filters to retrieved documents."""
        filtered_docs = []
        
        for doc in docs:
            metadata = doc.get("metadata", {})
            include_doc = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, list):
                        if metadata[filter_key] not in filter_value:
                            include_doc = False
                            break
                    else:
                        if metadata[filter_key] != filter_value:
                            include_doc = False
                            break
                else:
                    include_doc = False
                    break
            
            if include_doc:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG chain."""
        return {
            "config": {
                "retrieval_k": self.config.retrieval_k,
                "similarity_threshold": self.config.similarity_threshold,
                "max_context_length": self.config.max_context_length,
                "include_sources": self.config.include_sources
            },
            "indexer_stats": self.indexer.get_indexing_stats()
        }
    
    def update_config(self, **kwargs):
        """Update RAG chain configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown config parameter: {key}")
