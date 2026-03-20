"""
Unit tests for ingestion module.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker, Chunk
from ingestion.preprocessor import DocumentPreprocessor

class TestDocumentLoader(unittest.TestCase):
    """Test cases for DocumentLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DocumentLoader(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_txt_file(self):
        """Test loading a text file."""
        # Create a test text file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "This is a test document. It has multiple sentences."
        test_file.write_text(test_content)
        
        # Load the document
        documents = self.loader.load_documents([str(test_file)])
        
        # Assertions
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], test_content)
        self.assertEqual(documents[0]["metadata"]["file_name"], "test.txt")
        self.assertEqual(documents[0]["metadata"]["file_type"], ".txt")
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_documents(["nonexistent.txt"])
    
    def test_unsupported_file_format(self):
        """Test loading an unsupported file format."""
        # Create a test file with unsupported extension
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.write_text("content")
        
        with self.assertRaises(ValueError):
            self.loader.load_documents([str(test_file)])
    
    def test_get_all_documents(self):
        """Test getting all documents from directory."""
        # Create multiple test files
        (Path(self.temp_dir) / "test1.txt").write_text("Content 1")
        (Path(self.temp_dir) / "test2.txt").write_text("Content 2")
        (Path(self.temp_dir) / "subdir").mkdir()
        (Path(self.temp_dir) / "subdir" / "test3.txt").write_text("Content 3")
        
        # Load all documents
        documents = self.loader.load_documents()
        
        # Should find all text files
        self.assertEqual(len(documents), 3)

class TestDocumentChunker(unittest.TestCase):
    """Test cases for DocumentChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        self.test_document = {
            "content": "This is a test document. It has multiple sentences. " * 10,
            "metadata": {"source": "test.txt"}
        }
    
    def test_recursive_chunking(self):
        """Test recursive chunking strategy."""
        self.chunker.strategy = "recursive"
        chunks = self.chunker.chunk_single_document(self.test_document, 0)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIn("content", chunk.content)
            self.assertEqual(chunk.metadata["document_index"], 0)
            self.assertEqual(chunk.metadata["chunk_strategy"], "recursive")
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        self.chunker.strategy = "fixed_size"
        chunks = self.chunker.chunk_single_document(self.test_document, 0)
        
        # Should create chunks with specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), self.chunker.chunk_size + self.chunker.chunk_overlap)
    
    def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        self.chunker.strategy = "semantic"
        chunks = self.chunker.chunk_single_document(self.test_document, 0)
        
        # Should create chunks based on sentences
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
    
    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents."""
        documents = [
            {"content": "First document content.", "metadata": {"source": "doc1.txt"}},
            {"content": "Second document content.", "metadata": {"source": "doc2.txt"}}
        ]
        
        chunks = self.chunker.chunk_documents(documents)
        
        # Should create chunks for both documents
        self.assertGreater(len(chunks), 0)
        
        # Check document indices
        doc_indices = {chunk.metadata["document_index"] for chunk in chunks}
        self.assertEqual(doc_indices, {0, 1})

class TestDocumentPreprocessor(unittest.TestCase):
    """Test cases for DocumentPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DocumentPreprocessor()
        self.test_chunk = Chunk(
            content="This is a test chunk with URL: https://example.com and email: test@example.com",
            metadata={"source": "test.txt"},
            chunk_id="test_chunk_1",
            start_index=0,
            end_index=80
        )
    
    def test_remove_urls(self):
        """Test URL removal."""
        self.preprocessor.remove_urls = True
        processed_chunk = self.preprocessor.preprocess_chunks([self.test_chunk])[0]
        
        self.assertNotIn("https://example.com", processed_chunk.content)
    
    def test_remove_emails(self):
        """Test email removal."""
        self.preprocessor.remove_emails = True
        processed_chunk = self.preprocessor.preprocess_chunks([self.test_chunk])[0]
        
        self.assertNotIn("test@example.com", processed_chunk.content)
    
    def test_clean_whitespace(self):
        """Test whitespace cleaning."""
        chunk_with_extra_spaces = Chunk(
            content="This    has   extra   spaces",
            metadata={"source": "test.txt"},
            chunk_id="test_chunk_2",
            start_index=0,
            end_index=30
        )
        
        self.preprocessor.clean_whitespace = True
        processed_chunk = self.preprocessor.preprocess_chunks([chunk_with_extra_spaces])[0]
        
        self.assertEqual(processed_chunk.content, "This has extra spaces")
    
    def test_filter_chunks_by_length(self):
        """Test filtering chunks by length."""
        chunks = [
            Chunk(content="Short", metadata={}, chunk_id="1", start_index=0, end_index=5),
            Chunk(content="This is a medium length chunk", metadata={}, chunk_id="2", start_index=0, end_index=30),
            Chunk(content="This is a very long chunk that exceeds the maximum length limit for filtering", metadata={}, chunk_id="3", start_index=0, end_index=80)
        ]
        
        filtered = self.preprocessor.filter_chunks_by_length(chunks, min_length=10, max_length=50)
        
        # Should only keep the medium length chunk
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, "2")
    
    def test_remove_duplicate_chunks(self):
        """Test removing duplicate chunks."""
        chunks = [
            Chunk(content="Unique content 1", metadata={}, chunk_id="1", start_index=0, end_index=15),
            Chunk(content="Duplicate content", metadata={}, chunk_id="2", start_index=0, end_index=17),
            Chunk(content="Duplicate content", metadata={}, chunk_id="3", start_index=0, end_index=17),
            Chunk(content="Unique content 2", metadata={}, chunk_id="4", start_index=0, end_index=15)
        ]
        
        unique_chunks = self.preprocessor.remove_duplicate_chunks(chunks)
        
        # Should remove one duplicate
        self.assertEqual(len(unique_chunks), 3)
        chunk_ids = {chunk.chunk_id for chunk in unique_chunks}
        self.assertEqual(chunk_ids, {"1", "2", "4"})

if __name__ == "__main__":
    unittest.main()
