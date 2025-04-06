"""
Model factory for SmolaVision.
"""

import os
import logging
from typing import Dict, Any, Optional, Type

from smolavision.exceptions import ConfigurationError
from smolavision.models.base import ModelInterface
from smolavision.models.anthropic import AnthropicModel
from smolavision.models.openai import OpenAIModel
from smolavision.models.ollama import OllamaModel
from smolavision.models.huggingface import HuggingFaceModel
from smolavision.models.gemini import GeminiModel # Import the new GeminiModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating model instances."""
    
    # Registry of model types to their implementation classes
    _model_registry: Dict[str, Type[ModelInterface]] = {
        "anthropic": AnthropicModel,
        "openai": OpenAIModel,
        "ollama": OllamaModel,
        "huggingface": HuggingFaceModel,
        "gemini": GeminiModel # Add gemini to the registry
    }

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[ModelInterface]) -> None:
        """
        Register a new model implementation.
        
        Args:
            model_type: Model type identifier
            model_class: Model implementation class
        """
        cls._model_registry[model_type] = model_class
        logger.debug(f"Registered model type: {model_type}")

    @staticmethod
    def _get_api_key(config: Dict[str, Any], key_name: str) -> str:
        """
        Helper function to retrieve API key from config or environment.
        
        Args:
            config: Configuration dictionary
            key_name: Environment variable name for the API key
            
        Returns:
            API key
            
        Raises:
            ConfigurationError: If API key is not provided
        """
        api_key = config.get("api_key", os.environ.get(key_name))
        if not api_key:
            raise ConfigurationError(f"{key_name.replace('_API_KEY', '')} API key not provided")
        return api_key

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> ModelInterface:
        """
        Create a model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance

        Raises:
            ConfigurationError: If model creation fails
        """
        # Default to anthropic if not specified
        model_type = config.get("model_type", "anthropic")

        logger.info(f"Creating model of type: {model_type}")
        
        try:
            if model_type not in cls._model_registry:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
            
            model_class = cls._model_registry[model_type]
            
            if model_type == "anthropic":
                api_key = cls._get_api_key(config, "ANTHROPIC_API_KEY")
                return model_class(
                    api_key=api_key,
                    model_id=config.get("model_name", "claude-3-opus-20240229"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 4096)
                )
                
            elif model_type == "openai":
                api_key = cls._get_api_key(config, "OPENAI_API_KEY")
                return model_class(
                    api_key=api_key,
                    model_id=config.get("model_name", "gpt-4o"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 4096)
                )
                
            elif model_type == "ollama":
                ollama_config = config.get("ollama", {})
                return model_class(
                    base_url=ollama_config.get("base_url", "http://localhost:11434"),
                    model_name=ollama_config.get("model_name", "llama3"),
                    vision_model_name=ollama_config.get("vision_model", "llava"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 4096),
                    request_timeout=config.get("request_timeout", 60)
                )
                
            elif model_type == "huggingface":
                token = config.get("api_key", os.environ.get("HUGGINGFACE_API_KEY"))
                return model_class(
                    token=token,
                    model_id=config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct"),
                    text_pipeline_task=config.get("text_pipeline_task", "text-generation"),
                    vision_pipeline_task=config.get("vision_pipeline_task", "image-to-text"),
                    temperature=config.get("temperature", 0.7),
                    max_new_tokens=config.get("max_tokens", 4096)
                )

            elif model_type == "gemini":
                # Use GEMINI_API_KEY for LiteLLM
                api_key = cls._get_api_key(config, "GEMINI_API_KEY")
                # Pass model_id without 'gemini/' prefix, the class handles it
                model_id = config.get("model_name", GeminiModel.DEFAULT_MODEL.split('/')[-1])
                return model_class(
                    api_key=api_key,
                    model_id=model_id,
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 4096)
                    # Pass other config items if needed by GeminiModel.__init__ or LiteLLM
                )

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise ConfigurationError(f"Missing dependencies: {e}. Please ensure you've installed all dependencies") from e
        except Exception as e:
            logger.exception(f"Failed to create model instance")
            raise ConfigurationError(f"Failed to create model instance: {e}") from e

    @classmethod
    def create_vision_model(cls, config: Dict[str, Any]) -> ModelInterface:
        """
        Create a vision model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance for vision tasks

        Raises:
            ConfigurationError: If model creation fails
        """
        # Create a copy of the config to avoid modifying the original
        vision_config = dict(config)
        
        # Set the model name to the vision model
        if "vision_model" in config:
            vision_config["model_name"] = config["vision_model"]
            
        return cls.create_model(vision_config)

    @classmethod
    def create_summary_model(cls, config: Dict[str, Any]) -> ModelInterface:
        """
        Create a summary model instance based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A model instance for summarization

        Raises:
            ConfigurationError: If model creation fails
        """
        # Create a copy of the config to avoid modifying the original
        summary_config = dict(config)
        
        # Set the model name to the summary model
        if "summary_model" in config:
            summary_config["model_name"] = config["summary_model"]
            
        return cls.create_model(summary_config)
