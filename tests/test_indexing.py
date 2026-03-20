"""
Unit tests for indexing module.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.embeddings import (
    EmbeddingGenerator, OpenAIEmbeddingProvider, 
    HuggingFaceEmbeddingProvider, create_embedding_generator
)
from indexing.vector_store import (
    VectorStoreManager, VectorStoreConfig, ChromaVectorStore,
    InMemoryVectorStore
)
from indexing.indexer import DocumentIndexer

class TestEmbeddingProviders(unittest.TestCase):
    """Test cases for embedding providers."""
    
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-ada-002",
            api_key="test-key"
        )
        
        self.assertEqual(provider.model_name, "text-embedding-ada-002")
        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.get_embedding_dimension(), 1536)
    
    def test_huggingface_provider_initialization(self):
        """Test Hugging Face provider initialization."""
        provider = HuggingFaceEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.assertEqual(provider.model_name, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(provider.get_embedding_dimension(), 384)
    
    def test_embedding_generator_batch_processing(self):
        """Test embedding generator batch processing."""
        provider = Mock()
        provider.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
        provider.get_embedding_dimension.return_value = 2
        
        generator = EmbeddingGenerator(provider)
        
        texts = ["text1", "text2"]
        embeddings = generator.generate_embeddings(texts, batch_size=1)
        
        # Should call embed_texts twice due to batch_size=1
        self.assertEqual(provider.embed_texts.call_count, 2)
        self.assertEqual(len(embeddings), 2)
    
    def test_create_embedding_generator_factory(self):
        """Test embedding generator factory function."""
        # Test OpenAI provider creation
        generator = create_embedding_generator("openai", model_name="ada-002")
        self.assertIsInstance(generator.provider, OpenAIEmbeddingProvider)
        
        # Test Hugging Face provider creation
        generator = create_embedding_generator("huggingface", model_name="minilm")
        self.assertIsInstance(generator.provider, HuggingFaceEmbeddingProvider)
        
        # Test invalid provider
        with self.assertRaises(ValueError):
            create_embedding_generator("invalid")

class TestVectorStore(unittest.TestCase):
    """Test cases for vector store."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorStoreConfig(
            store_type="memory",
            persist_directory=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_in_memory_vector_store(self):
        """Test in-memory vector store."""
        store = InMemoryVectorStore(self.config)
        
        # Add documents
        docs = [
            {"id": "1", "content": "test1", "embedding": [0.1, 0.2], "metadata": {}},
            {"id": "2", "content": "test2", "embedding": [0.3, 0.4], "metadata": {}}
        ]
        
        ids = store.add_documents(docs)
        self.assertEqual(ids, ["1", "2"])
        
        # Search documents
        results = store.similarity_search([0.1, 0.2], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "1")  # Most similar should be first
    
    def test_vector_store_manager_initialization(self):
        """Test vector store manager initialization."""
        manager = VectorStoreManager(self.config)
        
        self.assertIsInstance(manager.vector_store, InMemoryVectorStore)
        self.assertEqual(manager.config.store_type, "memory")
    
    def test_vector_store_add_and_search(self):
        """Test adding and searching documents."""
        manager = VectorStoreManager(self.config)
        
        docs = [
            {"id": "1", "content": "test", "embedding": [0.1, 0.2], "metadata": {}}
        ]
        
        ids = manager.add_documents(docs)
        self.assertEqual(ids, ["1"])
        
        results = manager.search([0.1, 0.2])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")

class TestDocumentIndexer(unittest.TestCase):
    """Test cases for document indexer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock embedding generator
        self.mock_embedding_generator = Mock()
        self.mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]
        self.mock_embedding_generator.generate_query_embedding.return_value = [0.1, 0.2]
        
        # Vector store config
        self.vector_config = VectorStoreConfig(
            store_type="memory",
            persist_directory=self.temp_dir
        )
        
        self.indexer = DocumentIndexer(
            self.mock_embedding_generator,
            self.vector_config
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_index_chunks(self):
        """Test indexing chunks."""
        from ingestion.chunker import Chunk
        
        chunks = [
            Chunk(
                content="Test chunk content",
                metadata={"source": "test.txt"},
                chunk_id="test_1",
                start_index=0,
                end_index=18
            )
        ]
        
        result = self.indexer.index_chunks(chunks)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["chunks_processed"], 1)
        self.assertEqual(result["documents_indexed"], 1)
    
    def test_search_similar_chunks(self):
        """Test searching for similar chunks."""
        # First index some chunks
        from ingestion.chunker import Chunk
        
        chunks = [
            Chunk(
                content="Test chunk content",
                metadata={"source": "test.txt"},
                chunk_id="test_1",
                start_index=0,
                end_index=18
            )
        ]
        
        self.indexer.index_chunks(chunks)
        
        # Now search
        results = self.indexer.search_similar_chunks("test query")
        
        self.assertIsInstance(results, list)
    
    def test_get_indexing_stats(self):
        """Test getting indexing statistics."""
        stats = self.indexer.get_indexing_stats()
        
        self.assertIn("documents_indexed", stats)
        self.assertIn("chunks_indexed", stats)
        self.assertIn("vector_store", stats)
    
    def test_validate_index(self):
        """Test index validation."""
        validation = self.indexer.validate_index()
        
        self.assertIn("valid", validation)
        self.assertIn("vector_store_count", validation)
        self.assertIn("indexed_count", validation)

class TestIntegration(unittest.TestCase):
    """Integration tests for indexing module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_indexing(self):
        """Test end-to-end indexing process."""
        from ingestion.chunker import Chunk
        
        # Create mock embedding generator
        mock_generator = Mock()
        mock_generator.generate_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_generator.generate_query_embedding.return_value = [0.1, 0.2]
        
        # Create indexer
        config = VectorStoreConfig(store_type="memory", persist_directory=self.temp_dir)
        indexer = DocumentIndexer(mock_generator, config)
        
        # Create test chunks
        chunks = [
            Chunk(
                content="First test chunk",
                metadata={"source": "test1.txt"},
                chunk_id="test_1",
                start_index=0,
                end_index=16
            ),
            Chunk(
                content="Second test chunk",
                metadata={"source": "test2.txt"},
                chunk_id="test_2",
                start_index=0,
                end_index=17
            )
        ]
        
        # Index chunks
        result = indexer.index_chunks(chunks)
        self.assertTrue(result["success"])
        self.assertEqual(result["documents_indexed"], 2)
        
        # Search
        search_results = indexer.search_similar_chunks("test query")
        self.assertGreater(len(search_results), 0)
        
        # Get stats
        stats = indexer.get_indexing_stats()
        self.assertEqual(stats["documents_indexed"], 2)

if __name__ == "__main__":
    unittest.main()
