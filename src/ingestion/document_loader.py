"""
Document loader for various file types.
Supports PDF, TXT, DOCX, and other common document formats.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

class DocumentLoader:
    """Handles loading of various document types."""
    
    def __init__(self, data_dir: str = "data/source_documents"):
        self.data_dir = Path(data_dir)
        self.supported_formats = {'.pdf', '.txt', '.docx', '.md', '.html', '.json'}
    
    def load_documents(self, file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load documents from specified file paths or all documents in data directory.
        
        Args:
            file_paths: List of specific file paths to load. If None, loads all supported files.
            
        Returns:
            List of dictionaries containing document content and metadata.
        """
        documents = []
        
        if file_paths is None:
            file_paths = self._get_all_documents()
        
        for file_path in file_paths:
            try:
                doc = self._load_single_document(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _get_all_documents(self) -> List[str]:
        """Get all supported document files in the data directory."""
        documents = []
        for ext in self.supported_formats:
            documents.extend(self.data_dir.rglob(f"*{ext}"))
        return [str(doc) for doc in documents]
    
    def _load_single_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document based on its file type."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Implementation would depend on specific libraries
        # This is a placeholder structure
        content = self._extract_content(path)
        
        return {
            "content": content,
            "metadata": {
                "source": str(path),
                "file_type": path.suffix.lower(),
                "file_name": path.name,
                "file_size": path.stat().st_size
            }
        }
    
    def _extract_content(self, path: Path) -> str:
        """Extract text content from a document file."""
        # Placeholder implementation
        # In real implementation, use libraries like:
        # - PyPDF2 or pdfplumber for PDFs
        # - python-docx for DOCX
        # - markdown for MD files
        # - BeautifulSoup for HTML
        
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # For other formats, return placeholder
            return f"Content from {path.name} (extraction not implemented)"
