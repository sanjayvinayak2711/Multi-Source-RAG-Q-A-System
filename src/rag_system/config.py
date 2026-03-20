"""
RAG System Configuration
Configuration management for the RAG system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from models.model_config import ModelType, ProviderType

class RAGConfig:
    """RAG system configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/app_config.yaml"
        self.config = self._load_config()
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"Loaded configuration from {self.config_path}")
                return config
            else:
                print(f"Configuration file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "app": {
                "name": "Multi-Source RAG System",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO",
                "host": "0.0.0.0",
                "port": 8000
            },
            "database": {
                "type": "chroma",
                "persist_directory": "data/vector_store",
                "collection_name": "documents",
                "embedding_dimension": 1536
            },
            "models": {
                "embedding": {
                    "provider": "openai",
                    "model_name": "text-embedding-ada-002",
                    "api_key_env": "OPENAI_API_KEY",
                    "batch_size": 32,
                    "embedding_dimension": 1536
                },
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "api_key_env": "OPENAI_API_KEY",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            },
            "rag": {
                "retrieval": {
                    "top_k": 5,
                    "similarity_threshold": 0.7,
                    "search_type": "similarity"
                },
                "chunking": {
                    "strategy": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "min_chunk_length": 50,
                    "max_chunk_length": 2000
                },
                "context": {
                    "max_context_length": 4000,
                    "max_documents": 5,
                    "include_metadata": True,
                    "include_scores": True,
                    "context_format": "document"
                },
                "query_processing": {
                    "expand_queries": True,
                    "max_expansions": 3,
                    "remove_stopwords": False,
                    "normalize_text": True,
                    "min_query_length": 3
                }
            },
            "document_processing": {
                "supported_formats": [".pdf", ".txt", ".docx", ".md", ".html", ".json"],
                "preprocessing": {
                    "clean_whitespace": True,
                    "remove_urls": True,
                    "remove_emails": True,
                    "normalize_text": True,
                    "deduplicate": True
                },
                "data_directory": "data/source_documents",
                "watch_directory": False
            },
            "cache": {
                "enabled": True,
                "type": "redis",
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "password": None,
                    "db": 0,
                    "ttl": 3600
                },
                "memory": {
                    "max_size": 1000,
                    "ttl": 1800
                }
            },
            "evaluation": {
                "enabled": True,
                "metrics": {
                    "retrieval": ["hit_rate", "mrr", "precision_at_k", "ndcg_at_k"],
                    "generation": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                },
                "datasets": {
                    "test_data_path": "data/evaluation/test_data.json",
                    "validation_data_path": "data/evaluation/validation_data.json"
                }
            },
            "api": {
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                    "burst_size": 10
                },
                "authentication": {
                    "enabled": False,
                    "type": "api_key",
                    "api_key_env": "API_KEY"
                },
                "cors": {
                    "enabled": True,
                    "origins": ["*"],
                    "methods": ["GET", "POST", "PUT", "DELETE"],
                    "headers": ["*"]
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "prometheus": {
                        "enabled": True,
                        "port": 9090,
                        "endpoint": "/metrics"
                    }
                },
                "logging": {
                    "file_logging": True,
                    "log_file": "logs/app.log",
                    "log_rotation": "daily",
                    "max_file_size": "10MB",
                    "backup_count": 5
                },
                "health_check": {
                    "enabled": True,
                    "endpoint": "/health",
                    "interval": 30
                }
            },
            "security": {
                "input_validation": {
                    "max_query_length": 1000,
                    "max_document_size": 10485760,
                    "allowed_file_types": [".pdf", ".txt", ".docx", ".md", ".html", ".json"]
                },
                "output_filtering": {
                    "max_response_length": 10000,
                    "sanitize_html": True,
                    "remove_sensitive_data": True
                }
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            # App settings
            "ENV": ("app.debug", lambda x: x.lower() == "true"),
            "LOG_LEVEL": ("app.log_level", str),
            "HOST": ("app.host", str),
            "PORT": ("app.port", int),
            
            # Database settings
            "CHROMA_PERSIST_DIRECTORY": ("database.persist_directory", str),
            "EMBEDDING_DIMENSION": ("database.embedding_dimension", int),
            
            # Model settings
            "OPENAI_API_KEY": ("models.embedding.api_key", str),
            "ANTHROPIC_API_KEY": ("models.llm.api_key", str),
            "EMBEDDING_MODEL": ("models.embedding.model_name", str),
            "LLM_MODEL": ("models.llm.model_name", str),
            
            # RAG settings
            "TOP_K": ("rag.retrieval.top_k", int),
            "SIMILARITY_THRESHOLD": ("rag.retrieval.similarity_threshold", float),
            "CHUNK_SIZE": ("rag.chunking.chunk_size", int),
            "CHUNK_OVERLAP": ("rag.chunking.chunk_overlap", int),
            "MAX_CONTEXT_LENGTH": ("rag.context.max_context_length", int),
            
            # Data directory
            "DATA_DIRECTORY": ("document_processing.data_directory", str),
            
            # Cache settings
            "REDIS_HOST": ("cache.redis.host", str),
            "REDIS_PORT": ("cache.redis.port", int),
            "REDIS_PASSWORD": ("cache.redis.password", str),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_value(config_path, converted_value)
                    print(f"Applied environment override: {config_path} = {converted_value}")
                except (ValueError, TypeError) as e:
                    print(f"Invalid environment variable {env_var}={value}: {e}")
    
    def _set_nested_value(self, path: str, value: Any):
        """Set nested configuration value."""
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Dot-separated path to configuration value
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """
        Set configuration value by path.
        
        Args:
            path: Dot-separated path to configuration value
            value: Value to set
        """
        self._set_nested_value(path, value)
    
    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, uses original path.
        """
        save_path = output_path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Validate required settings
        if not self.get("models.embedding.api_key"):
            errors.append("Embedding API key not configured")
        
        if not self.get("models.llm.api_key"):
            errors.append("LLM API key not configured")
        
        # Validate data directory
        data_dir = Path(self.get("document_processing.data_directory"))
        if not data_dir.exists():
            warnings.append(f"Data directory {data_dir} does not exist")
        
        # Validate ranges
        top_k = self.get("rag.retrieval.top_k")
        if top_k is not None and (top_k < 1 or top_k > 100):
            errors.append("top_k must be between 1 and 100")
        
        chunk_size = self.get("rag.chunking.chunk_size")
        if chunk_size is not None and (chunk_size < 100 or chunk_size > 10000):
            errors.append("chunk_size must be between 100 and 10000")
        
        # Validate file formats
        supported_formats = self.get("document_processing.supported_formats", [])
        if not supported_formats:
            errors.append("No supported file formats configured")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_model_config(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration by type and name.
        
        Args:
            model_type: Type of model (embedding, llm, reranker)
            model_name: Name of the model
            
        Returns:
            Model configuration
        """
        # Load models configuration
        models_config_path = "config/models_config.yaml"
        try:
            with open(models_config_path, 'r', encoding='utf-8') as f:
                models_config = yaml.safe_load(f)
            
            model_key = f"{model_type}_models"
            if model_key in models_config and model_name in models_config[model_key]:
                return models_config[model_key][model_name]
            else:
                return {}
        except Exception as e:
            print(f"Error loading models configuration: {e}")
            return {}
    
    def get_prompt_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get prompt template by name.
        
        Args:
            template_name: Name of the prompt template
            
        Returns:
            Prompt template configuration
        """
        # Load prompts configuration
        prompts_config_path = "config/prompts_config.yaml"
        try:
            with open(prompts_config_path, 'r', encoding='utf-8') as f:
                prompts_config = yaml.safe_load(f)
            
            # Search in all prompt categories
            for category in prompts_config.values():
                if isinstance(category, dict) and template_name in category:
                    return category[template_name]
            
            return {}
        except Exception as e:
            print(f"Error loading prompts configuration: {e}")
            return {}
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"RAGConfig(path={self.config_path}, keys={list(self.config.keys())})"
