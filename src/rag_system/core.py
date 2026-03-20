"""
Core RAG System Implementation
Main orchestrator for the RAG pipeline.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ingestion import DocumentLoader, DocumentChunker, DocumentPreprocessor
from indexing import DocumentIndexer, EmbeddingGenerator, VectorStoreConfig
from chains import RAGChain, QueryProcessor, ContextBuilder
from models import ModelManager, ModelConfig
from prompts import PromptManager
from evaluation import RAGEvaluator

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.evaluator = RAGEvaluator()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize document processing
        self.document_loader = DocumentLoader(
            self.config.get("data_directory", "data/source_documents")
        )
        
        self.document_chunker = DocumentChunker(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            strategy=self.config.get("chunking_strategy", "recursive")
        )
        
        self.document_preprocessor = DocumentPreprocessor(
            clean_whitespace=self.config.get("clean_whitespace", True),
            remove_urls=self.config.get("remove_urls", True),
            remove_emails=self.config.get("remove_emails", True),
            normalize_text=self.config.get("normalize_text", True)
        )
        
        # Initialize indexing
        embedding_config = self._get_embedding_config()
        self.embedding_generator = self.model_manager.get_embedding_generator(
            embedding_config["name"]
        )
        
        vector_config = VectorStoreConfig(
            store_type=self.config.get("vector_store", "chroma"),
            persist_directory=self.config.get("persist_directory", "data/vector_store"),
            embedding_dimension=embedding_config.get("embedding_dimension", 1536)
        )
        
        self.document_indexer = DocumentIndexer(
            self.embedding_generator, vector_config
        )
        
        # Initialize chains
        self.query_processor = QueryProcessor(
            expand_queries=self.config.get("expand_queries", True),
            max_expansions=self.config.get("max_expansions", 3)
        )
        
        self.context_builder = ContextBuilder(
            max_context_length=self.config.get("max_context_length", 4000),
            max_documents=self.config.get("max_documents", 5)
        )
        
        self.rag_chain = RAGChain(
            indexer=self.document_indexer,
            llm_client=self._get_llm_client(),
            config=self._get_rag_config()
        )
    
    def _get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration."""
        embedding_name = self.config.get("embedding_model", "openai-embedding-ada")
        config = self.model_manager.get_model_config(embedding_name)
        
        if not config:
            raise ValueError(f"Embedding model '{embedding_name}' not found")
        
        return {
            "name": config.name,
            "provider": config.provider,
            "model_id": config.model_id,
            "embedding_dimension": getattr(config, 'embedding_dimension', 1536)
        }
    
    def _get_llm_client(self):
        """Get LLM client."""
        llm_name = self.config.get("llm_model", "openai-gpt-35-turbo")
        config = self.model_manager.get_model_config(llm_name)
        
        if not config:
            raise ValueError(f"LLM model '{llm_name}' not found")
        
        # This would create actual LLM client based on provider
        # For now, return a mock client
        return MockLLMClient(config)
    
    def _get_rag_config(self):
        """Get RAG chain configuration."""
        from chains.rag_chain import RAGConfig
        
        return RAGConfig(
            retrieval_k=self.config.get("top_k", 5),
            similarity_threshold=self.config.get("similarity_threshold", 0.7),
            max_context_length=self.config.get("max_context_length", 4000),
            include_sources=self.config.get("include_sources", True),
            include_scores=self.config.get("include_scores", True)
        )
    
    async def index_documents(self, file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Index documents into the RAG system.
        
        Args:
            file_paths: List of file paths to index. If None, index all documents.
            
        Returns:
            Dictionary with indexing results
        """
        try:
            logger.info("Starting document indexing...")
            
            # Load documents
            documents = self.document_loader.load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Chunk documents
            chunks = self.document_chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Preprocess chunks
            processed_chunks = self.document_preprocessor.preprocess_chunks(chunks)
            logger.info(f"Processed {len(processed_chunks)} chunks")
            
            # Index chunks
            result = self.document_indexer.index_chunks(processed_chunks)
            logger.info(f"Indexed {result.get('documents_indexed', 0)} chunks")
            
            return {
                "success": True,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "chunks_indexed": result.get("documents_indexed", 0),
                "message": "Documents indexed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to index documents"
            }
    
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Query question
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary with query results
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Process query
            processed_query = self.query_processor.process_query(question)
            
            # Get answer using RAG chain
            result = self.rag_chain.answer_question(
                question,
                context_filter=kwargs.get("context_filter")
            )
            
            # Add query processing metadata
            result["query_processing"] = {
                "original_query": processed_query["original_query"],
                "processed_query": processed_query["processed_query"],
                "expanded_queries": processed_query["expanded_queries"]
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process query"
            }
    
    async def evaluate_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system performance.
        
        Args:
            test_data: List of test examples
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"Evaluating system with {len(test_data)} test examples...")
            
            results = []
            
            for example in test_data:
                # Process query
                query_result = await self.query(
                    example["question"],
                    context_filter=example.get("context_filter")
                )
                
                # Prepare evaluation data
                eval_data = {
                    "question": example["question"],
                    "answer": query_result.get("answer", ""),
                    "context": query_result.get("context", ""),
                    "retrieved_docs": query_result.get("sources", []),
                    "relevant_docs": example.get("relevant_docs", [])
                }
                
                results.append(eval_data)
            
            # Evaluate results
            evaluation_results = self.evaluator.evaluate_dataset(results)
            
            logger.info("System evaluation completed")
            return {
                "success": True,
                "evaluation_results": evaluation_results,
                "num_examples": len(test_data),
                "message": "System evaluation completed"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating system: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate system"
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            indexing_stats = self.document_indexer.get_indexing_stats()
            model_stats = self.model_manager.get_config_summary()
            
            return {
                "indexing": indexing_stats,
                "models": model_stats,
                "config": {
                    "version": "1.0.0",
                    "components_initialized": True
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            # Check components
            components_status = {
                "document_loader": True,
                "document_chunker": True,
                "document_preprocessor": True,
                "embedding_generator": True,
                "vector_store": True,
                "rag_chain": True
            }
            
            # Check system stats
            stats = self.get_system_stats()
            
            # Overall health
            healthy = all(components_status.values()) and "error" not in stats
            
            return {
                "healthy": healthy,
                "components": components_status,
                "stats": stats,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }

class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def generate(self, prompt: str) -> str:
        """Generate response for testing."""
        return f"Mock response based on prompt: {prompt[:100]}..."
