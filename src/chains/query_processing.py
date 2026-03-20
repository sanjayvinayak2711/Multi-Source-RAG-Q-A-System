"""
Query processing module for RAG system.
Handles query preprocessing, expansion, and optimization.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class QueryConfig:
    """Configuration for query processing."""
    expand_queries: bool = True
    max_expansions: int = 3
    remove_stopwords: bool = False
    normalize_text: bool = True
    min_query_length: int = 3

class QueryProcessor:
    """Handles query preprocessing and expansion."""
    
    def __init__(self, config: QueryConfig = None):
        self.config = config or QueryConfig()
        self.stopwords = self._load_stopwords()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query with various techniques.
        
        Args:
            query: Original query string
            
        Returns:
            Dictionary with processed query and metadata
        """
        if not query or len(query.strip()) < self.config.min_query_length:
            return {
                "original_query": query,
                "processed_query": query,
                "expanded_queries": [],
                "metadata": {"error": "Query too short"}
            }
        
        # Step 1: Basic preprocessing
        processed_query = self._preprocess_query(query)
        
        # Step 2: Query expansion
        expanded_queries = []
        if self.config.expand_queries:
            expanded_queries = self._expand_query(processed_query)
        
        return {
            "original_query": query,
            "processed_query": processed_query,
            "expanded_queries": expanded_queries,
            "metadata": {
                "query_length": len(query),
                "processed_length": len(processed_query),
                "expansion_count": len(expanded_queries)
            }
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Apply basic preprocessing to query."""
        processed = query.strip()
        
        if self.config.normalize_text:
            processed = self._normalize_text(processed)
        
        if self.config.remove_stopwords:
            processed = self._remove_stopwords(processed)
        
        return processed
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        expansions = []
        
        # Method 1: Synonym expansion (placeholder)
        synonym_expansions = self._get_synonyms(query)
        expansions.extend(synonym_expansions)
        
        # Method 2: N-gram expansion
        ngram_expansions = self._get_ngram_expansions(query)
        expansions.extend(ngram_expansions)
        
        # Method 3: Question reformulation
        reformulations = self._reformulate_question(query)
        expansions.extend(reformulations)
        
        # Limit number of expansions
        return expansions[:self.config.max_expansions]
    
    def _get_synonyms(self, query: str) -> List[str]:
        """Get synonym expansions for query."""
        # Placeholder implementation
        # In real implementation, use WordNet, spaCy, or other NLP libraries
        
        synonyms = []
        words = query.split()
        
        # Simple synonym mapping example
        synonym_map = {
            "how": "what way",
            "what": "which",
            "where": "what place",
            "when": "what time",
            "why": "what reason",
            "explain": "describe",
            "define": "explain",
            "show": "demonstrate",
            "tell": "inform"
        }
        
        for word in words:
            if word.lower() in synonym_map:
                expanded_query = query.replace(word, synonym_map[word.lower()], 1)
                synonyms.append(expanded_query)
        
        return synonyms
    
    def _get_ngram_expansions(self, query: str) -> List[str]:
        """Get n-gram based expansions."""
        expansions = []
        words = query.split()
        
        # Add bigram expansions
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                expansions.append(bigram)
        
        # Add trigram expansions
        if len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                expansions.append(trigram)
        
        return expansions
    
    def _reformulate_question(self, query: str) -> List[str]:
        """Reformulate question in different ways."""
        reformulations = []
        
        # Add question word variations
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        
        for qw in question_words:
            if not query.lower().startswith(qw):
                reformulation = f"{qw} {query}"
                reformulations.append(reformulation)
        
        # Add "tell me about" prefix
        if not query.lower().startswith("tell me"):
            reformulations.append(f"tell me about {query}")
        
        # Add "information about" suffix
        if not query.lower().endswith("information"):
            reformulations.append(f"{query} information")
        
        return reformulations
    
    def _load_stopwords(self) -> set:
        """Load stopwords set."""
        # Basic English stopwords
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'about', 'up', 'out', 'many', 'then', 'them', 'can', 'may',
            'after', 'before', 'could', 'should', 'would', 'been', 'being',
            'did', 'does', 'do', 'am', 'i', 'you', 'we', 'us', 'our', 'your'
        }
    
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract entities from query.
        
        Args:
            query: Query string
            
        Returns:
            List of extracted entities
        """
        # Placeholder implementation
        # In real implementation, use spaCy or other NER libraries
        
        entities = []
        
        # Simple pattern matching for common entities
        patterns = {
            "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "NUMBER": r'\b\d+(?:\.\d+)?\b',
            "URL": r'https?://[^\s]+',
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": entity_type,
                    "start": query.find(match),
                    "end": query.find(match) + len(match)
                })
        
        return entities
    
    def classify_query_type(self, query: str) -> str:
        """
        Classify the type of query.
        
        Args:
            query: Query string
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Question patterns
        if any(qw in query_lower for qw in ["what", "how", "why", "when", "where", "which", "who"]):
            return "question"
        
        # Command patterns
        if any(cmd in query_lower for cmd in ["show", "tell", "explain", "describe", "find", "search"]):
            return "command"
        
        # Information request
        if any(info in query_lower for info in ["information", "details", "about", "regarding"]):
            return "information"
        
        # Default
        return "general"
