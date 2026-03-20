"""
RAG System API
FastAPI-based REST API for the RAG system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .core import RAGSystem
from .config import RAGConfig

logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of documents to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_sources: Optional[bool] = Field(True, description="Include source information")
    expand_query: Optional[bool] = Field(True, description="Expand query for better retrieval")

class IndexRequest(BaseModel):
    """Request model for document indexing."""
    file_paths: Optional[List[str]] = Field(None, description="List of file paths to index")
    reindex: Optional[bool] = Field(False, description="Reindex all documents")

class EvaluationRequest(BaseModel):
    """Request model for system evaluation."""
    test_data_path: Optional[str] = Field(None, description="Path to test data file")
    metrics: Optional[List[str]] = Field(None, description="Metrics to evaluate")

class QueryResponse(BaseModel):
    """Response model for queries."""
    success: bool
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float
    query_processing: Optional[Dict[str, Any]] = None
    retrieved_count: int
    context_length: int
    error: Optional[str] = None

class IndexResponse(BaseModel):
    """Response model for indexing."""
    success: bool
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    message: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health checks."""
    healthy: bool
    components: Dict[str, bool]
    stats: Optional[Dict[str, Any]] = None
    timestamp: float
    error: Optional[str] = None

class RAGAPI:
    """FastAPI application for RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = RAGConfig(config_path)
        
        # Initialize RAG system
        self.rag_system = RAGSystem(self.config.config)
        
        # Initialize FastAPI
        self.app = FastAPI(
            title=self.config.get("app.name", "Multi-Source RAG System"),
            version=self.config.get("app.version", "1.0.0"),
            description="A comprehensive RAG system with multi-source document support",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("RAG API initialized successfully")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        cors_config = self.config.get("api.cors", {})
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=cors_config.get("methods", ["*"]),
            allow_headers=cors_config.get("headers", ["*"])
        )
        
        # Add custom middleware for logging
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            logger.info(f"Request: {request.method} {request.url}")
            response = await call_next(request)
            logger.info(f"Response: {response.status_code}")
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", tags=["Root"])
        async def root():
            """Root endpoint."""
            return {
                "message": "Multi-Source RAG System API",
                "version": self.config.get("app.version", "1.0.0"),
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check():
            """Health check endpoint."""
            try:
                health_data = await self.rag_system.health_check()
                return HealthResponse(**health_data)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthResponse(
                    healthy=False,
                    components={},
                    timestamp=0,
                    error=str(e)
                )
        
        @self.app.post("/query", response_model=QueryResponse, tags=["Query"])
        async def query_rag(request: QueryRequest):
            """Query the RAG system."""
            try:
                # Prepare query parameters
                query_params = {
                    "top_k": request.top_k,
                    "similarity_threshold": request.similarity_threshold,
                    "include_sources": request.include_sources,
                    "expand_query": request.expand_query
                }
                
                # Process query
                result = await self.rag_system.query(request.question, **query_params)
                
                return QueryResponse(**result)
                
            except Exception as e:
                logger.error(f"Query error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/index", response_model=IndexResponse, tags=["Indexing"])
        async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
            """Index documents into the RAG system."""
            try:
                # Run indexing in background
                background_tasks.add_task(self._index_documents_background, request)
                
                return IndexResponse(
                    success=True,
                    documents_loaded=0,
                    chunks_created=0,
                    chunks_indexed=0,
                    message="Indexing started in background"
                )
                
            except Exception as e:
                logger.error(f"Indexing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/upload", response_model=IndexResponse, tags=["Upload"])
        async def upload_documents(
            files: List[UploadFile] = File(..., description="Documents to upload and index"),
            background_tasks: BackgroundTasks
        ):
            """Upload and index documents."""
            try:
                # Validate file types
                allowed_types = self.config.get("security.input_validation.allowed_file_types", [])
                max_size = self.config.get("security.input_validation.max_document_size", 10 * 1024 * 1024)
                
                uploaded_files = []
                total_size = 0
                
                for file in files:
                    # Check file type
                    file_ext = Path(file.filename).suffix.lower()
                    if file_ext not in allowed_types:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"File type {file_ext} not allowed"
                        )
                    
                    # Check file size
                    file_size = await file.read()
                    total_size += len(file_size)
                    if total_size > max_size:
                        raise HTTPException(
                            status_code=400,
                            detail="File size exceeds maximum limit"
                        )
                    
                    uploaded_files.append(file_size)
                
                # Save files and index
                background_tasks.add_task(self._process_uploaded_files, uploaded_files)
                
                return IndexResponse(
                    success=True,
                    documents_loaded=len(uploaded_files),
                    chunks_created=0,
                    chunks_indexed=0,
                    message="Files uploaded and indexing started"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Upload error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/evaluate", tags=["Evaluation"])
        async def evaluate_system(request: EvaluationRequest):
            """Evaluate the RAG system."""
            try:
                # Load test data
                test_data = []
                if request.test_data_path:
                    import json
                    with open(request.test_data_path, 'r') as f:
                        test_data = json.load(f)
                
                # Run evaluation
                result = await self.rag_system.evaluate_system(test_data)
                
                return result
                
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats", tags=["System"])
        async def get_system_stats():
            """Get system statistics."""
            try:
                stats = self.rag_system.get_system_stats()
                return stats
            except Exception as e:
                logger.error(f"Stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config", tags=["Configuration"])
        async def get_config():
            """Get system configuration (non-sensitive parts only)."""
            try:
                # Return non-sensitive configuration
                safe_config = {
                    "app": self.config.get("app"),
                    "rag": self.config.get("rag"),
                    "document_processing": {
                        "supported_formats": self.config.get("document_processing.supported_formats")
                    },
                    "models": {
                        "embedding": {
                            "model_name": self.config.get("models.embedding.model_name"),
                            "provider": self.config.get("models.embedding.provider")
                        },
                        "llm": {
                            "model_name": self.config.get("models.llm.model_name"),
                            "provider": self.config.get("models.llm.provider")
                        }
                    }
                }
                return safe_config
            except Exception as e:
                logger.error(f"Config error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _index_documents_background(self, request: IndexRequest):
        """Background task for document indexing."""
        try:
            result = await self.rag_system.index_documents(request.file_paths)
            logger.info(f"Background indexing completed: {result}")
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")
    
    async def _process_uploaded_files(self, files: List[bytes]):
        """Background task for processing uploaded files."""
        try:
            # Save files to data directory
            data_dir = Path(self.config.get("document_processing.data_directory"))
            data_dir.mkdir(parents=True, exist_ok=True)
            
            file_paths = []
            for i, file_content in enumerate(files):
                file_path = data_dir / f"uploaded_{i}.txt"
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                file_paths.append(str(file_path))
            
            # Index the files
            result = await self.rag_system.index_documents(file_paths)
            logger.info(f"Uploaded files processed: {result}")
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Run the FastAPI application.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        # Get configuration values
        host = host or self.config.get("app.host", "0.0.0.0")
        port = port or self.config.get("app.port", 8000)
        debug = debug if debug is not None else self.config.get("app.debug", False)
        
        # Configure logging
        log_level = self.config.get("app.log_level", "INFO")
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        
        logger.info(f"Starting RAG API on {host}:{port} (debug={debug})")
        
        # Run uvicorn
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=debug,
            access_log=True
        )

# Create global API instance
api_instance = None

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI application instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application instance
    """
    global api_instance
    if api_instance is None:
        api_instance = RAGAPI(config_path)
    return api_instance.app

def run_server(config_path: Optional[str] = None, **kwargs):
    """
    Run the RAG API server.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional server configuration
    """
    global api_instance
    if api_instance is None:
        api_instance = RAGAPI(config_path)
    
    api_instance.run(**kwargs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Source RAG System API")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
        debug=args.debug
    )
