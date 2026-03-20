"""
Model manager for RAG system.
Handles model loading, configuration, and lifecycle management.
"""

from typing import Dict, Any, Optional, List, Type
import os
import json
from datetime import datetime
from .model_config import (
    ModelConfig, EmbeddingModelConfig, LLMConfig, RerankerModelConfig,
    PredefinedModels, ModelType, ProviderType
)

class ModelManager:
    """Manages model configurations and instances."""
    
    def __init__(self, config_dir: str = "models/configs"):
        self.config_dir = config_dir
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_instances: Dict[str, Any] = {}
        self.predefined_models = PredefinedModels()
        self._ensure_config_dir()
        self._load_predefined_configs()
        self._load_custom_configs()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists."""
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _load_predefined_configs(self):
        """Load predefined model configurations."""
        predefined = self.predefined_models.get_all_predefined()
        self.model_configs.update(predefined)
    
    def _load_custom_configs(self):
        """Load custom model configurations from files."""
        config_files = [
            "embedding_models.json",
            "llm_models.json",
            "reranker_models.json"
        ]
        
        for config_file in config_files:
            file_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        configs_data = json.load(f)
                    
                    for name, config_data in configs_data.items():
                        config = self._create_config_from_data(config_data)
                        if config:
                            self.model_configs[name] = config
                            
                except Exception as e:
                    print(f"Error loading config from {config_file}: {e}")
    
    def _create_config_from_data(self, data: Dict[str, Any]) -> Optional[ModelConfig]:
        """Create model configuration from data dictionary."""
        try:
            model_type = ModelType(data.get("model_type", "llm"))
            
            if model_type == ModelType.EMBEDDING:
                return EmbeddingModelConfig.from_dict(data)
            elif model_type == ModelType.LLM:
                return LLMConfig.from_dict(data)
            elif model_type == ModelType.RERANKER:
                return RerankerModelConfig.from_dict(data)
            else:
                return ModelConfig.from_dict(data)
                
        except Exception as e:
            print(f"Error creating config from data: {e}")
            return None
    
    def add_model_config(self, config: ModelConfig) -> bool:
        """
        Add a new model configuration.
        
        Args:
            config: Model configuration to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model_configs[config.name] = config
            self._save_config_to_file(config)
            return True
        except Exception as e:
            print(f"Error adding model config: {e}")
            return False
    
    def _save_config_to_file(self, config: ModelConfig):
        """Save configuration to appropriate file."""
        if isinstance(config, EmbeddingModelConfig):
            filename = "embedding_models.json"
        elif isinstance(config, LLMConfig):
            filename = "llm_models.json"
        elif isinstance(config, RerankerModelConfig):
            filename = "reranker_models.json"
        else:
            filename = "other_models.json"
        
        file_path = os.path.join(self.config_dir, filename)
        
        # Load existing configs
        existing_configs = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_configs = json.load(f)
            except Exception:
                existing_configs = {}
        
        # Add new config
        existing_configs[config.name] = config.to_dict()
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(existing_configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config to file: {e}")
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.model_configs.get(name)
    
    def list_model_configs(self, model_type: Optional[ModelType] = None, 
                          provider: Optional[ProviderType] = None) -> List[str]:
        """
        List available model configurations.
        
        Args:
            model_type: Filter by model type
            provider: Filter by provider
            
        Returns:
            List of model configuration names
        """
        configs = []
        
        for name, config in self.model_configs.items():
            if model_type and config.model_type != model_type:
                continue
            if provider and config.provider != provider:
                continue
            configs.append(name)
        
        return configs
    
    def get_embedding_models(self) -> List[str]:
        """Get list of available embedding model names."""
        return self.list_model_configs(model_type=ModelType.EMBEDDING)
    
    def get_llm_models(self) -> List[str]:
        """Get list of available LLM model names."""
        return self.list_model_configs(model_type=ModelType.LLM)
    
    def get_reranker_models(self) -> List[str]:
        """Get list of available reranker model names."""
        return self.list_model_configs(model_type=ModelType.RERANKER)
    
    def get_models_by_provider(self, provider: ProviderType) -> List[str]:
        """Get list of models from a specific provider."""
        return self.list_model_configs(provider=provider)
    
    def remove_model_config(self, name: str) -> bool:
        """
        Remove a model configuration.
        
        Args:
            name: Name of the configuration to remove
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.model_configs:
            return False
        
        # Don't allow removing predefined models
        if name in self.predefined_models.get_all_predefined():
            print("Cannot remove predefined model configuration")
            return False
        
        config = self.model_configs[name]
        del self.model_configs[name]
        
        # Remove from file
        if isinstance(config, EmbeddingModelConfig):
            filename = "embedding_models.json"
        elif isinstance(config, LLMConfig):
            filename = "llm_models.json"
        elif isinstance(config, RerankerModelConfig):
            filename = "reranker_models.json"
        else:
            filename = "other_models.json"
        
        file_path = os.path.join(self.config_dir, filename)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    configs_data = json.load(f)
                
                if name in configs_data:
                    del configs_data[name]
                    
                    with open(file_path, 'w') as f:
                        json.dump(configs_data, f, indent=2)
                        
            except Exception as e:
                print(f"Error removing config from file: {e}")
        
        # Remove from instances if loaded
        if name in self.model_instances:
            del self.model_instances[name]
        
        return True
    
    def validate_config(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Validate a model configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Basic validation
        if not config.name:
            errors.append("Model name is required")
        
        if not config.model_id:
            errors.append("Model ID is required")
        
        # Provider-specific validation
        if config.provider in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.COHERE]:
            if not config.api_key:
                warnings.append("API key not provided - will need to be set via environment")
        
        if config.provider == ProviderType.LOCAL:
            if not config.api_base:
                warnings.append("Local API base URL not provided")
        
        # Model type specific validation
        if isinstance(config, EmbeddingModelConfig):
            if config.embedding_dimension <= 0:
                errors.append("Embedding dimension must be positive")
            if config.batch_size <= 0:
                errors.append("Batch size must be positive")
        
        elif isinstance(config, LLMConfig):
            if config.max_tokens <= 0:
                errors.append("Max tokens must be positive")
            if not 0 <= config.temperature <= 2:
                errors.append("Temperature must be between 0 and 2")
            if not 0 <= config.top_p <= 1:
                errors.append("Top-p must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model configuration."""
        config = self.get_model_config(name)
        if not config:
            return None
        
        info = config.to_dict()
        info["is_predefined"] = name in self.predefined_models.get_all_predefined()
        info["is_loaded"] = name in self.model_instances
        
        # Add provider-specific information
        if config.provider == ProviderType.OPENAI:
            info["provider_info"] = {
                "api_docs": "https://platform.openai.com/docs/api-reference",
                "pricing_url": "https://openai.com/pricing"
            }
        elif config.provider == ProviderType.HUGGINGFACE:
            info["provider_info"] = {
                "api_docs": "https://huggingface.co/docs/api-inference",
                "model_url": f"https://huggingface.co/{config.model_id}"
            }
        elif config.provider == ProviderType.ANTHROPIC:
            info["provider_info"] = {
                "api_docs": "https://docs.anthropic.com/claude/reference",
                "pricing_url": "https://www.anthropic.com/pricing"
            }
        
        return info
    
    def export_configs(self, file_path: str, include_predefined: bool = False) -> bool:
        """
        Export model configurations to a file.
        
        Args:
            file_path: Path to export file
            include_predefined: Whether to include predefined configs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            configs_to_export = {}
            
            for name, config in self.model_configs.items():
                if include_predefined or name not in self.predefined_models.get_all_predefined():
                    configs_to_export[name] = config.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(configs_to_export, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting configs: {e}")
            return False
    
    def import_configs(self, file_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Import model configurations from a file.
        
        Args:
            file_path: Path to import file
            overwrite: Whether to overwrite existing configs
            
        Returns:
            Import result
        """
        try:
            with open(file_path, 'r') as f:
                configs_data = json.load(f)
            
            imported = []
            skipped = []
            errors = []
            
            for name, config_data in configs_data.items():
                if name in self.model_configs and not overwrite:
                    skipped.append(name)
                    continue
                
                try:
                    config = self._create_config_from_data(config_data)
                    if config:
                        self.add_model_config(config)
                        imported.append(name)
                    else:
                        errors.append(f"Failed to create config for {name}")
                        
                except Exception as e:
                    errors.append(f"Error importing {name}: {e}")
            
            return {
                "imported": imported,
                "skipped": skipped,
                "errors": errors
            }
            
        except Exception as e:
            return {"errors": [f"Error reading file: {e}"]}
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all model configurations."""
        summary = {
            "total_configs": len(self.model_configs),
            "by_type": {},
            "by_provider": {},
            "loaded_instances": len(self.model_instances)
        }
        
        for config in self.model_configs.values():
            # Count by type
            type_name = config.model_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Count by provider
            provider_name = config.provider.value
            summary["by_provider"][provider_name] = summary["by_provider"].get(provider_name, 0) + 1
        
        return summary
