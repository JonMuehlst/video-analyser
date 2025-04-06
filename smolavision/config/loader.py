import os
import json
import logging
from typing import Dict, Any, Optional
from argparse import Namespace

from smolavision.config.schema import Config, VideoConfig, ModelConfig, AnalysisConfig, OllamaConfig
from smolavision.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")

def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Configuration dictionary with values from environment variables
    """
    config = {}
    
    # Model configuration
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        config["model"] = config.get("model", {})
        config["model"]["model_type"] = "anthropic"
        config["model"]["api_key"] = api_key
    elif api_key := os.environ.get("OPENAI_API_KEY"):
        config["model"] = config.get("model", {})
        config["model"]["model_type"] = "openai"
        config["model"]["api_key"] = api_key
    elif api_key := os.environ.get("HUGGINGFACE_API_KEY"):
        config["model"] = config.get("model", {})
        config["model"]["model_type"] = "huggingface"
        config["model"]["api_key"] = api_key
    elif api_key := os.environ.get("GEMINI_API_KEY"):
        config["model"] = config.get("model", {})
        config["model"]["model_type"] = "gemini"
        config["model"]["api_key"] = api_key

    # Ollama configuration (keep separate from API key logic)
    if os.environ.get("OLLAMA_ENABLED") == "true":
        # Ensure model section exists if only Ollama env vars are set
        config["model"] = config.get("model", {})
        # Only set model_type to ollama if no other API key forced a different type
        if "model_type" not in config["model"]:
             config["model"]["model_type"] = "ollama"
        # Store Ollama specific settings
        config["model"]["model_type"] = "ollama"
        config["model"]["ollama"] = config["model"].get("ollama", {})
        config["model"]["ollama"]["enabled"] = True
        
        if base_url := os.environ.get("OLLAMA_BASE_URL"):
            config["model"]["ollama"]["base_url"] = base_url
        if model_name := os.environ.get("OLLAMA_MODEL_NAME"):
            config["model"]["ollama"]["model_name"] = model_name
        if vision_model := os.environ.get("OLLAMA_VISION_MODEL"):
            config["model"]["ollama"]["vision_model"] = vision_model
    
    return config

def load_config_from_args(args: Namespace) -> Dict[str, Any]:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary with values from arguments
    """
    config = {}
    
    # Video configuration
    if hasattr(args, "language"):
        config["video"] = config.get("video", {})
        config["video"]["language"] = args.language
    if hasattr(args, "frame_interval"):
        config["video"] = config.get("video", {})
        config["video"]["frame_interval"] = args.frame_interval
    if hasattr(args, "detect_scenes"):
        config["video"] = config.get("video", {})
        config["video"]["detect_scenes"] = args.detect_scenes
    if hasattr(args, "scene_threshold"):
        config["video"] = config.get("video", {})
        config["video"]["scene_threshold"] = args.scene_threshold
    if hasattr(args, "enable_ocr"):
        config["video"] = config.get("video", {})
        config["video"]["enable_ocr"] = args.enable_ocr
    if hasattr(args, "start_time"):
        config["video"] = config.get("video", {})
        config["video"]["start_time"] = args.start_time
    if hasattr(args, "end_time"):
        config["video"] = config.get("video", {})
        config["video"]["end_time"] = args.end_time
    
    # Model configuration
    model_args = {}
    if hasattr(args, "model_type") and args.model_type is not None:
        model_args["model_type"] = args.model_type
    if hasattr(args, "api_key") and args.api_key is not None:
        model_args["api_key"] = args.api_key
    if hasattr(args, "vision_model") and args.vision_model is not None:
        model_args["vision_model"] = args.vision_model
    if hasattr(args, "summary_model") and args.summary_model is not None:
        model_args["summary_model"] = args.summary_model
    if model_args:
        config["model"] = {**config.get("model", {}), **model_args}
    
    # Ollama configuration
    if hasattr(args, "ollama_enabled") and args.ollama_enabled:
        config["model"] = config.get("model", {})
        config["model"]["model_type"] = "ollama"
        config["model"]["ollama"] = config["model"].get("ollama", {})
        config["model"]["ollama"]["enabled"] = True
        
        if hasattr(args, "ollama_base_url"):
            config["model"]["ollama"]["base_url"] = args.ollama_base_url
        if hasattr(args, "ollama_model"):
            config["model"]["ollama"]["model_name"] = args.ollama_model
        if hasattr(args, "ollama_vision_model"):
            config["model"]["ollama"]["vision_model"] = args.ollama_vision_model
    
    # Analysis configuration
    if hasattr(args, "mission"):
        config["analysis"] = config.get("analysis", {})
        config["analysis"]["mission"] = args.mission
    if hasattr(args, "generate_flowchart"):
        config["analysis"] = config.get("analysis", {})
        config["analysis"]["generate_flowchart"] = args.generate_flowchart
    if hasattr(args, "max_batch_size_mb"):
        config["analysis"] = config.get("analysis", {})
        config["analysis"]["max_batch_size_mb"] = args.max_batch_size_mb
    if hasattr(args, "max_images_per_batch"):
        config["analysis"] = config.get("analysis", {})
        config["analysis"]["max_images_per_batch"] = args.max_images_per_batch
    
    # Output directory
    if hasattr(args, "output_dir"):
        config["output_dir"] = args.output_dir
    
    return config

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        for section, section_config in config.items():
            if section not in result:
                result[section] = {}
            
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    if isinstance(value, dict) and key in result[section] and isinstance(result[section][key], dict):
                        # Merge nested dictionaries
                        result[section][key] = {**result[section][key], **value}
                    else:
                        # Override or add value
                        result[section][key] = value
            else:
                result[section] = section_config
    
    return result

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return Config().to_dict()

def load_config(config_path: Optional[str] = None, args: Optional[Namespace] = None) -> Dict[str, Any]:
    """
    Load configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Command line arguments
    2. Environment variables
    3. Configuration file
    4. Default values
    
    Args:
        config_path: Path to the configuration file (optional)
        args: Command line arguments (optional)
        
    Returns:
        Merged configuration dictionary
    """
    # Start with default configuration
    config = create_default_config()
    
    # Load from file if provided
    if config_path:
        file_config = load_config_from_file(config_path)
        config = merge_configs(config, file_config)
    
    # Load from environment variables
    env_config = load_config_from_env()
    config = merge_configs(config, env_config)
    
    # Load from command line arguments if provided
    if args:
        args_config = load_config_from_args(args)
        config = merge_configs(config, args_config)
    
    return config
