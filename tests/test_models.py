"""
Unit tests for models module.
"""

import unittest
import tempfile
import os
import json
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_config import (
    ModelConfig, EmbeddingModelConfig, LLMConfig, RerankerModelConfig,
    PredefinedModels, ModelType, ProviderType
)
from models.model_manager import ModelManager

class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LLM,
            provider=ProviderType.OPENAI,
            model_id="gpt-4"
        )
        
        self.assertEqual(config.name, "test-model")
        self.assertEqual(config.model_type, ModelType.LLM)
        self.assertEqual(config.provider, ProviderType.OPENAI)
        self.assertEqual(config.model_id, "gpt-4")
    
    def test_model_config_to_dict(self):
        """Test ModelConfig to dictionary conversion."""
        config = ModelConfig(
            name="test-model",
            model_type=ModelType.EMBEDDING,
            provider=ProviderType.HUGGINGFACE,
            model_id="bert-base",
            api_key="test-key"
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["name"], "test-model")
        self.assertEqual(config_dict["model_type"], "embedding")
        self.assertEqual(config_dict["provider"], "huggingface")
        self.assertEqual(config_dict["model_id"], "bert-base")
        self.assertEqual(config_dict["api_key"], "test-key")
    
    def test_model_config_from_dict(self):
        """Test ModelConfig from dictionary creation."""
        config_dict = {
            "name": "test-model",
            "model_type": "llm",
            "provider": "openai",
            "model_id": "gpt-3.5-turbo",
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        self.assertEqual(config.name, "test-model")
        self.assertEqual(config.model_type, ModelType.LLM)
        self.assertEqual(config.provider, ProviderType.OPENAI)
        self.assertEqual(config.model_id, "gpt-3.5-turbo")
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.temperature, 0.7)

class TestEmbeddingModelConfig(unittest.TestCase):
    """Test cases for EmbeddingModelConfig."""
    
    def test_embedding_config_creation(self):
        """Test EmbeddingModelConfig creation."""
        config = EmbeddingModelConfig(
            name="test-embedding",
            provider=ProviderType.OPENAI,
            model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            batch_size=32
        )
        
        self.assertEqual(config.embedding_dimension, 1536)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.model_type, ModelType.EMBEDDING)
    
    def test_embedding_config_defaults(self):
        """Test EmbeddingModelConfig default values."""
        config = EmbeddingModelConfig(
            name="test-embedding",
            provider=ProviderType.HUGGINGFACE,
            model_id="bert-base"
        )
        
        self.assertEqual(config.embedding_dimension, 1536)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize_embeddings)
    
    def test_embedding_config_to_dict(self):
        """Test EmbeddingModelConfig to dictionary conversion."""
        config = EmbeddingModelConfig(
            name="test-embedding",
            provider=ProviderType.OPENAI,
            model_id="text-embedding-ada-002",
            embedding_dimension=768,
            batch_size=16
        )
        
        config_dict = config.to_dict()
        
        self.assertIn("embedding_dimension", config_dict)
        self.assertIn("batch_size", config_dict)
        self.assertIn("normalize_embeddings", config_dict)
        self.assertEqual(config_dict["embedding_dimension"], 768)
        self.assertEqual(config_dict["batch_size"], 16)

class TestLLMConfig(unittest.TestCase):
    """Test cases for LLMConfig."""
    
    def test_llm_config_creation(self):
        """Test LLMConfig creation."""
        config = LLMConfig(
            name="test-llm",
            provider=ProviderType.OPENAI,
            model_id="gpt-4",
            max_tokens=4096,
            temperature=0.5,
            top_p=0.9
        )
        
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.model_type, ModelType.LLM)
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig(
            name="test-llm",
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3"
        )
        
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 1.0)
        self.assertFalse(config.stream)
    
    def test_llm_config_validation(self):
        """Test LLMConfig parameter validation."""
        # Valid config
        config = LLMConfig(
            name="test-llm",
            provider=ProviderType.OPENAI,
            model_id="gpt-4",
            temperature=0.7,
            top_p=0.9
        )
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)

class TestRerankerModelConfig(unittest.TestCase):
    """Test cases for RerankerModelConfig."""
    
    def test_reranker_config_creation(self):
        """Test RerankerModelConfig creation."""
        config = RerankerModelConfig(
            name="test-reranker",
            provider=ProviderType.COHERE,
            model_id="rerank-english-v2.0",
            top_k=5,
            score_threshold=0.8
        )
        
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.score_threshold, 0.8)
        self.assertEqual(config.model_type, ModelType.RERANKER)

class TestPredefinedModels(unittest.TestCase):
    """Test cases for PredefinedModels."""
    
    def test_get_all_predefined(self):
        """Test getting all predefined models."""
        all_models = PredefinedModels.get_all_predefined()
        
        self.assertIn("openai-gpt-4", all_models)
        self.assertIn("openai-embedding-ada", all_models)
        self.assertIn("hf-minilm", all_models)
        self.assertIn("anthropic-claude-3-sonnet", all_models)
    
    def test_get_embedding_models(self):
        """Test getting embedding models only."""
        embedding_models = PredefinedModels.get_embedding_models()
        
        self.assertIn("openai-embedding-ada", embedding_models)
        self.assertIn("hf-minilm", embedding_models)
        self.assertIn("hf-bge-small", embedding_models)
        
        # Should not contain LLM models
        self.assertNotIn("openai-gpt-4", embedding_models)
    
    def test_get_llm_models(self):
        """Test getting LLM models only."""
        llm_models = PredefinedModels.get_llm_models()
        
        self.assertIn("openai-gpt-4", llm_models)
        self.assertIn("anthropic-claude-3-sonnet", llm_models)
        self.assertIn("local-llama-7b", llm_models)
        
        # Should not contain embedding models
        self.assertNotIn("openai-embedding-ada", llm_models)
    
    def test_get_models_by_provider(self):
        """Test getting models by provider."""
        openai_models = PredefinedModels.get_models_by_provider(ProviderType.OPENAI)
        
        self.assertIn("openai-gpt-4", openai_models)
        self.assertIn("openai-embedding-ada", openai_models)
        
        hf_models = PredefinedModels.get_models_by_provider(ProviderType.HUGGINGFACE)
        
        self.assertIn("hf-minilm", hf_models)
        self.assertIn("hf-bge-base", hf_models)

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test ModelManager initialization."""
        self.assertIsInstance(self.manager.model_configs, dict)
        self.assertGreater(len(self.manager.model_configs), 0)  # Should have predefined models
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        config = self.manager.get_model_config("openai-gpt-4")
        
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "openai-gpt-4")
        self.assertEqual(config.provider, ProviderType.OPENAI)
    
    def test_get_nonexistent_model_config(self):
        """Test getting non-existent model configuration."""
        config = self.manager.get_model_config("nonexistent-model")
        self.assertIsNone(config)
    
    def test_list_model_configs(self):
        """Test listing model configurations."""
        all_configs = self.manager.list_model_configs()
        self.assertGreater(len(all_configs), 0)
        
        embedding_configs = self.manager.list_model_configs(model_type=ModelType.EMBEDDING)
        self.assertGreater(len(embedding_configs), 0)
        
        openai_configs = self.manager.list_model_configs(provider=ProviderType.OPENAI)
        self.assertGreater(len(openai_configs), 0)
    
    def test_add_model_config(self):
        """Test adding a new model configuration."""
        new_config = EmbeddingModelConfig(
            name="test-embedding",
            provider=ProviderType.HUGGINGFACE,
            model_id="test-bert",
            embedding_dimension=512
        )
        
        result = self.manager.add_model_config(new_config)
        self.assertTrue(result)
        
        # Verify it was added
        retrieved_config = self.manager.get_model_config("test-embedding")
        self.assertIsNotNone(retrieved_config)
        self.assertEqual(retrieved_config.name, "test-embedding")
    
    def test_remove_model_config(self):
        """Test removing a model configuration."""
        # Add a custom config first
        new_config = LLMConfig(
            name="test-llm",
            provider=ProviderType.LOCAL,
            model_id="test-model"
        )
        self.manager.add_model_config(new_config)
        
        # Remove it
        result = self.manager.remove_model_config("test-llm")
        self.assertTrue(result)
        
        # Verify it was removed
        retrieved_config = self.manager.get_model_config("test-llm")
        self.assertIsNone(retrieved_config)
    
    def test_remove_predefined_model_config(self):
        """Test that predefined models cannot be removed."""
        result = self.manager.remove_model_config("openai-gpt-4")
        self.assertFalse(result)
        
        # Verify it still exists
        config = self.manager.get_model_config("openai-gpt-4")
        self.assertIsNotNone(config)
    
    def test_validate_config(self):
        """Test model configuration validation."""
        # Valid config
        valid_config = LLMConfig(
            name="valid-llm",
            provider=ProviderType.OPENAI,
            model_id="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2048
        )
        
        validation = self.manager.validate_config(valid_config)
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)
        
        # Invalid config (missing model_id)
        invalid_config = LLMConfig(
            name="invalid-llm",
            provider=ProviderType.OPENAI,
            model_id=""  # Empty model_id
        )
        
        validation = self.manager.validate_config(invalid_config)
        self.assertFalse(validation["valid"])
        self.assertGreater(len(validation["errors"]), 0)
    
    def test_get_model_info(self):
        """Test getting detailed model information."""
        info = self.manager.get_model_info("openai-gpt-4")
        
        self.assertIsNotNone(info)
        self.assertIn("name", info)
        self.assertIn("model_type", info)
        self.assertIn("provider", info)
        self.assertIn("is_predefined", info)
        self.assertTrue(info["is_predefined"])
    
    def test_export_import_configs(self):
        """Test exporting and importing configurations."""
        # Add a custom config
        custom_config = EmbeddingModelConfig(
            name="custom-embedding",
            provider=ProviderType.HUGGINGFACE,
            model_id="custom-bert"
        )
        self.manager.add_model_config(custom_config)
        
        # Export configs
        export_file = os.path.join(self.temp_dir, "exported_configs.json")
        result = self.manager.export_configs(export_file, include_predefined=False)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(export_file))
        
        # Create new manager and import
        new_manager = ModelManager(config_dir=os.path.join(self.temp_dir, "new"))
        import_result = new_manager.import_configs(export_file)
        
        self.assertIn("imported", import_result)
        self.assertIn("custom-embedding", import_result["imported"])
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        summary = self.manager.get_config_summary()
        
        self.assertIn("total_configs", summary)
        self.assertIn("by_type", summary)
        self.assertIn("by_provider", summary)
        self.assertIn("loaded_instances", summary)
        
        self.assertGreater(summary["total_configs"], 0)
        self.assertIn("embedding", summary["by_type"])
        self.assertIn("llm", summary["by_type"])

if __name__ == "__main__":
    unittest.main()
