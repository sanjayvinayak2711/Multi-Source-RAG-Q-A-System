"""
Pytest configuration and fixtures for RAG system tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.preprocessor import DocumentPreprocessor
from indexing.embeddings import EmbeddingGenerator, HuggingFaceEmbeddingProvider
from indexing.vector_store import VectorStoreManager, VectorStoreConfig
from indexing.indexer import DocumentIndexer
from models.model_manager import ModelManager
from evaluation.metrics import RAGEvaluator

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = Path(temp_dir) / "sample.txt"
    content = """
    This is a sample document for testing the RAG system.
    It contains multiple sentences and paragraphs.
    
    The document discusses various topics including artificial intelligence,
    machine learning, and natural language processing.
    
    This content will be used to test document loading, chunking,
    and other preprocessing functions.
    """
    file_path.write_text(content.strip())
    return str(file_path)

@pytest.fixture
def sample_document():
    """Create a sample document dictionary."""
    return {
        "content": "This is a test document with multiple sentences. It should be long enough to test chunking strategies effectively.",
        "metadata": {
            "source": "test.txt",
            "file_type": ".txt",
            "file_name": "test.txt"
        }
    }

@pytest.fixture
def document_loader(temp_dir):
    """Create a DocumentLoader instance."""
    return DocumentLoader(temp_dir)

@pytest.fixture
def document_chunker():
    """Create a DocumentChunker instance."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def document_preprocessor():
    """Create a DocumentPreprocessor instance."""
    return DocumentPreprocessor(
        clean_whitespace=True,
        remove_urls=True,
        remove_emails=True,
        normalize_text=True
    )

@pytest.fixture
def embedding_provider():
    """Create a mock embedding provider."""
    import numpy as np
    
    class MockEmbeddingProvider:
        def embed_texts(self, texts):
            return [np.random.rand(384).tolist() for _ in texts]
        
        def embed_query(self, text):
            return np.random.rand(384).tolist()
        
        def get_embedding_dimension(self):
            return 384
    
    return MockEmbeddingProvider()

@pytest.fixture
def embedding_generator(embedding_provider):
    """Create an EmbeddingGenerator instance."""
    return EmbeddingGenerator(embedding_provider)

@pytest.fixture
def vector_store_config(temp_dir):
    """Create a VectorStoreConfig instance."""
    return VectorStoreConfig(
        store_type="memory",
        persist_directory=temp_dir,
        embedding_dimension=384
    )

@pytest.fixture
def vector_store_manager(vector_store_config):
    """Create a VectorStoreManager instance."""
    return VectorStoreManager(vector_store_config)

@pytest.fixture
def document_indexer(embedding_generator, vector_store_config):
    """Create a DocumentIndexer instance."""
    return DocumentIndexer(embedding_generator, vector_store_config)

@pytest.fixture
def model_manager(temp_dir):
    """Create a ModelManager instance."""
    return ModelManager(config_dir=temp_dir)

@pytest.fixture
def rag_evaluator():
    """Create a RAGEvaluator instance."""
    return RAGEvaluator()

@pytest.fixture
def sample_dataset():
    """Create a sample evaluation dataset."""
    return [
        {
            "question": "What is artificial intelligence?",
            "answer": "Artificial intelligence is the simulation of human intelligence in machines.",
            "context": "AI refers to the simulation of human intelligence in machines that are programmed to think and learn.",
            "retrieved_docs": [
                {"content": "AI is the simulation of human intelligence.", "id": "doc1"},
                {"content": "Machines can be programmed to think.", "id": "doc2"}
            ],
            "relevant_docs": ["doc1"]
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "context": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
            "retrieved_docs": [
                {"content": "ML enables systems to learn from data.", "id": "doc3"},
                {"content": "Systems can improve from experience.", "id": "doc4"}
            ],
            "relevant_docs": ["doc3", "doc4"]
        }
    ]

@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    from ingestion.chunker import Chunk
    
    return [
        Chunk(
            content="This is the first test chunk.",
            metadata={"source": "test1.txt", "chunk_index": 0},
            chunk_id="test_1_0",
            start_index=0,
            end_index=28
        ),
        Chunk(
            content="This is the second test chunk.",
            metadata={"source": "test2.txt", "chunk_index": 0},
            chunk_id="test_2_0",
            start_index=0,
            end_index=29
        )
    ]

# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Custom assertions
def assert_chunk_structure(chunk):
    """Assert that a chunk has the correct structure."""
    assert hasattr(chunk, 'content')
    assert hasattr(chunk, 'metadata')
    assert hasattr(chunk, 'chunk_id')
    assert hasattr(chunk, 'start_index')
    assert hasattr(chunk, 'end_index')
    assert isinstance(chunk.content, str)
    assert isinstance(chunk.metadata, dict)
    assert isinstance(chunk.chunk_id, str)
    assert isinstance(chunk.start_index, int)
    assert isinstance(chunk.end_index, int)

def assert_document_structure(document):
    """Assert that a document has the correct structure."""
    assert isinstance(document, dict)
    assert 'content' in document
    assert 'metadata' in document
    assert isinstance(document['content'], str)
    assert isinstance(document['metadata'], dict)

def assert_embedding_structure(embedding):
    """Assert that an embedding has the correct structure."""
    assert isinstance(embedding, list)
    assert all(isinstance(x, (int, float)) for x in embedding)
    assert len(embedding) > 0

def assert_retrieval_result_structure(result):
    """Assert that a retrieval result has the correct structure."""
    assert isinstance(result, dict)
    assert 'id' in result
    assert 'content' in result
    assert 'score' in result
    assert 'metadata' in result
    assert isinstance(result['score'], (int, float))
    assert 0 <= result['score'] <= 1
