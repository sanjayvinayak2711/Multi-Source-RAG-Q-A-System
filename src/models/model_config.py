"""
Model configurations for RAG system.
Contains configuration classes for different model types.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json

class ModelType(Enum):
    """Model type enumeration."""
    EMBEDDING = "embedding"
    LLM = "llm"
    RERANKER = "reranker"

class ProviderType(Enum):
    """Provider type enumeration."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    LOCAL = "local"

@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str
    model_type: ModelType
    provider: ProviderType
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: int = 30
    retry_attempts: int = 3
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "provider": self.provider.value,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "additional_params": self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            provider=ProviderType(data["provider"]),
            model_id=data["model_id"],
            api_key=data.get("api_key"),
            api_base=data.get("api_base"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            timeout=data.get("timeout", 30),
            retry_attempts=data.get("retry_attempts", 3),
            additional_params=data.get("additional_params", {})
        )

@dataclass
class EmbeddingModelConfig(ModelConfig):
    """Configuration for embedding models."""
    embedding_dimension: int = 1536
    batch_size: int = 32
    normalize_embeddings: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.model_type = ModelType.EMBEDDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with embedding-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModelConfig":
        """Create from dictionary with embedding-specific fields."""
        base_config = super().from_dict(data)
        return cls(
            name=base_config.name,
            model_type=base_config.model_type,
            provider=base_config.provider,
            model_id=base_config.model_id,
            api_key=base_config.api_key,
            api_base=base_config.api_base,
            max_tokens=base_config.max_tokens,
            temperature=base_config.temperature,
            timeout=base_config.timeout,
            retry_attempts=base_config.retry_attempts,
            additional_params=base_config.additional_params,
            embedding_dimension=data.get("embedding_dimension", 1536),
            batch_size=data.get("batch_size", 32),
            normalize_embeddings=data.get("normalize_embeddings", True)
        )

@dataclass
class LLMConfig(ModelConfig):
    """Configuration for Large Language Models."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.model_type = ModelType.LLM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with LLM-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create from dictionary with LLM-specific fields."""
        base_config = super().from_dict(data)
        return cls(
            name=base_config.name,
            model_type=base_config.model_type,
            provider=base_config.provider,
            model_id=base_config.model_id,
            api_key=base_config.api_key,
            api_base=base_config.api_base,
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k"),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            stop_sequences=data.get("stop_sequences"),
            stream=data.get("stream", False),
            timeout=base_config.timeout,
            retry_attempts=base_config.retry_attempts,
            additional_params=base_config.additional_params
        )

@dataclass
class RerankerModelConfig(ModelConfig):
    """Configuration for reranker models."""
    top_k: int = 10
    score_threshold: float = 0.5
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.model_type = ModelType.RERANKER
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with reranker-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerModelConfig":
        """Create from dictionary with reranker-specific fields."""
        base_config = super().from_dict(data)
        return cls(
            name=base_config.name,
            model_type=base_config.model_type,
            provider=base_config.provider,
            model_id=base_config.model_id,
            api_key=base_config.api_key,
            api_base=base_config.api_base,
            top_k=data.get("top_k", 10),
            score_threshold=data.get("score_threshold", 0.5),
            timeout=base_config.timeout,
            retry_attempts=base_config.retry_attempts,
            additional_params=base_config.additional_params
        )

# Predefined model configurations
class PredefinedModels:
    """Collection of predefined model configurations."""
    
    # OpenAI Models
    OPENAI_EMBEDDING_ADA = EmbeddingModelConfig(
        name="openai-embedding-ada",
        provider=ProviderType.OPENAI,
        model_id="text-embedding-ada-002",
        embedding_dimension=1536,
        batch_size=100
    )
    
    OPENAI_EMBEDDING_SMALL = EmbeddingModelConfig(
        name="openai-embedding-small",
        provider=ProviderType.OPENAI,
        model_id="text-embedding-3-small",
        embedding_dimension=1536,
        batch_size=100
    )
    
    OPENAI_EMBEDDING_LARGE = EmbeddingModelConfig(
        name="openai-embedding-large",
        provider=ProviderType.OPENAI,
        model_id="text-embedding-3-large",
        embedding_dimension=3072,
        batch_size=50
    )
    
    OPENAI_GPT_35_TURBO = LLMConfig(
        name="openai-gpt-35-turbo",
        provider=ProviderType.OPENAI,
        model_id="gpt-3.5-turbo",
        max_tokens=2048,
        temperature=0.7
    )
    
    OPENAI_GPT_4 = LLMConfig(
        name="openai-gpt-4",
        provider=ProviderType.OPENAI,
        model_id="gpt-4",
        max_tokens=4096,
        temperature=0.7
    )
    
    OPENAI_GPT_4_TURBO = LLMConfig(
        name="openai-gpt-4-turbo",
        provider=ProviderType.OPENAI,
        model_id="gpt-4-turbo-preview",
        max_tokens=4096,
        temperature=0.7
    )
    
    # Hugging Face Models
    HF_MINILM = EmbeddingModelConfig(
        name="hf-minilm",
        provider=ProviderType.HUGGINGFACE,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        batch_size=32
    )
    
    HF_MPNET = EmbeddingModelConfig(
        name="hf-mpnet",
        provider=ProviderType.HUGGINGFACE,
        model_id="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=768,
        batch_size=16
    )
    
    HF_BGE_SMALL = EmbeddingModelConfig(
        name="hf-bge-small",
        provider=ProviderType.HUGGINGFACE,
        model_id="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        batch_size=32
    )
    
    HF_BGE_BASE = EmbeddingModelConfig(
        name="hf-bge-base",
        provider=ProviderType.HUGGINGFACE,
        model_id="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        batch_size=16
    )
    
    # Anthropic Models
    ANTHROPIC_CLAUDE_3_SONNET = LLMConfig(
        name="anthropic-claude-3-sonnet",
        provider=ProviderType.ANTHROPIC,
        model_id="claude-3-sonnet-20240229",
        max_tokens=4096,
        temperature=0.7
    )
    
    ANTHROPIC_CLAUDE_3_HAIKU = LLMConfig(
        name="anthropic-claude-3-haiku",
        provider=ProviderType.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0.7
    )
    
    # Local Models (placeholder configurations)
    LOCAL_LLAMA_7B = LLMConfig(
        name="local-llama-7b",
        provider=ProviderType.LOCAL,
        model_id="llama-7b-chat",
        max_tokens=2048,
        temperature=0.7,
        api_base="http://localhost:8000"
    )
    
    LOCAL_MISTRAL_7B = LLMConfig(
        name="local-mistral-7b",
        provider=ProviderType.LOCAL,
        model_id="mistral-7b-instruct",
        max_tokens=2048,
        temperature=0.7,
        api_base="http://localhost:8001"
    )
    
    @classmethod
    def get_all_predefined(cls) -> Dict[str, ModelConfig]:
        """Get all predefined model configurations."""
        return {
            # Embedding models
            "openai-embedding-ada": cls.OPENAI_EMBEDDING_ADA,
            "openai-embedding-small": cls.OPENAI_EMBEDDING_SMALL,
            "openai-embedding-large": cls.OPENAI_EMBEDDING_LARGE,
            "hf-minilm": cls.HF_MINILM,
            "hf-mpnet": cls.HF_MPNET,
            "hf-bge-small": cls.HF_BGE_SMALL,
            "hf-bge-base": cls.HF_BGE_BASE,
            
            # LLM models
            "openai-gpt-35-turbo": cls.OPENAI_GPT_35_TURBO,
            "openai-gpt-4": cls.OPENAI_GPT_4,
            "openai-gpt-4-turbo": cls.OPENAI_GPT_4_TURBO,
            "anthropic-claude-3-sonnet": cls.ANTHROPIC_CLAUDE_3_SONNET,
            "anthropic-claude-3-haiku": cls.ANTHROPIC_CLAUDE_3_HAIKU,
            "local-llama-7b": cls.LOCAL_LLAMA_7B,
            "local-mistral-7b": cls.LOCAL_MISTRAL_7B,
        }
    
    @classmethod
    def get_embedding_models(cls) -> Dict[str, EmbeddingModelConfig]:
        """Get all predefined embedding model configurations."""
        all_models = cls.get_all_predefined()
        return {
            name: config for name, config in all_models.items()
            if isinstance(config, EmbeddingModelConfig)
        }
    
    @classmethod
    def get_llm_models(cls) -> Dict[str, LLMConfig]:
        """Get all predefined LLM configurations."""
        all_models = cls.get_all_predefined()
        return {
            name: config for name, config in all_models.items()
            if isinstance(config, LLMConfig)
        }
    
    @classmethod
    def get_models_by_provider(cls, provider: ProviderType) -> Dict[str, ModelConfig]:
        """Get all models from a specific provider."""
        all_models = cls.get_all_predefined()
        return {
            name: config for name, config in all_models.items()
            if config.provider == provider
        }
