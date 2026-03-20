"""
Multi-Source RAG System Package
A comprehensive RAG (Retrieval-Augmented Generation) system with multi-source document support.
"""

__version__ = "1.0.0"
__author__ = "RAG System Team"
__email__ = "team@example.com"
__description__ = "A comprehensive RAG system with multi-source document support"

# Import main components for easy access
from . import ingestion
from . import indexing
from . import chains
from . import prompts
from . import models
from . import evaluation

__all__ = [
    "ingestion",
    "indexing", 
    "chains",
    "prompts",
    "models",
    "evaluation",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]
