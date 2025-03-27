"""
Model factory for SmolaVision.
"""

import os
from typing import Dict, Any, Optional

from ..exceptions import ConfigurationError
from ..logging import get_logger
from .base import ModelInterface
from .anthropic import AnthropicModel
from .openai import OpenAIModel
from .ollama import OllamaModel
from .huggingface import HuggingFaceModel

logger = get_logger("models.factory")


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def _get_api_key(config: Dict[str, Any], key_name: str) -> str:
        """Helper function to retrieve API key from config or environment."""
        api_key = config.get("api_key", os.environ.get(key_name))
        if not api_key:
            raise ConfigurationError(f"{key_name.replace('_API_KEY', '')} API key not provided")
        return api_key

    @staticmethod
    def _create_model_instance(model_type: str, config: Dict[str, Any], **kwargs) -> ModelInterface:
        """Helper function to create the model and handle exceptions."""
        try:
            if model_type == "anthropic":
                api_key = ModelFactory._get_api_key(config, "ANTHROPIC_API_KEY")
                return AnthropicModel(api_key=api_key, model_id=kwargs.get("model_name", config.get("model_name", "claude-3-opus-20240229")),
                                      temperature=config.get("temperature", 0.7), max_tokens=config.get("max_tokens", 4096))
            elif model_type == "openai":
                api_key = ModelFactory._get_api_key(config, "OPENAI_API_KEY")
                return OpenAIModel(api_key=api_key, model_id=kwargs.get("model_name", config.get("model_name", "gpt-4o")), temperature=config.get("temperature", 0.7), max_tokens=config.get("max_tokens", 4096))
            elif model_type == "ollama":
                ollama_config = config.get("ollama", {})
                return OllamaModel(base_url=ollama_config.get("base_url", "http://localhost:11434"), model_name=kwargs.get("model_name", ollama_config.get("model_name", "llama3")),
                                   vision_model_name=kwargs.get("vision_model_name", ollama_config.get("vision_model", "llava")), temperature=config.get("temperature", 0.7), max_tokens=config.get("max_tokens", 4096), request_timeout = config.get("request_timeout", 60))
            elif model_type == "huggingface":
                token = config.get("api_key", os.environ.get("HUGGINGFACE_API_KEY"))
                model_id = config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct")
                text_pipeline_task = config.get("text_pipeline_task", "text-generation")
                vision_pipeline_task = config.get("vision_pipeline_task", "image-to-text")
                temperature = config.get("temperature", 0.7) #Add temperature
                max_new_tokens = config.get("max_tokens", 4096) #And max tokens

                return HuggingFaceModel(token=token, model_id=model_id, text_pipeline_task=text_pipeline_task, vision_pipeline_task=vision_pipeline_task, temperature = temperature, max_new_tokens = max_new_tokens)

            else:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
        except ImportError as e:
            raise ConfigurationError(f"Missing dependencies: {e}. Please ensure you've installed all dependencies") from None
        except Exception as e:
            raise ConfigurationError(f"Failed to create model instance: {e}") from None

    @staticmethod
    def create_model(config: Dict[str, Any]) -> ModelInterface:
        """
        Create a model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance

        Raises:
            ConfigurationError: If model creation fails
        """
        return ModelFactory._create_model_instance(config.get("model_type", "anthropic"), config)

    @staticmethod
    def create_vision_model(config: Dict[str, Any]) -> ModelInterface:
        """
        Create a vision model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance for vision tasks

        Raises:
            ConfigurationError: If model creation fails
        """
        model_type = config.get("model_type", "anthropic")  # Consistent model_type
        return ModelFactory._create_model_instance(model_type, config, model_name=config.get("vision_model"))

    @staticmethod
    def create_summary_model(config: Dict[str, Any]) -> ModelInterface:
        """
        Create a summary model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance for summarization

        Raises:
            ConfigurationError: If model creation fails
        """
        model_type = config.get("model_type", "anthropic")  # Consistent model_type
        return ModelFactory._create_model_instance(model_type, config, model_name=config.get("summary_model"))