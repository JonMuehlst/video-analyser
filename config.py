"""
Configuration module for SmolaVision
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class OllamaConfig:
    """Configuration for Ollama models"""
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    vision_model: str = "llava"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
@dataclass
class ModelConfig:
    """Configuration for AI models"""
    api_key: str = ""
    model_type: str = "anthropic"  # anthropic, openai, huggingface, ollama
    vision_model: str = "claude"
    summary_model: str = "claude-3-5-sonnet-20240620"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result["ollama"] = self.ollama.to_dict()
        return result

@dataclass
class VideoConfig:
    """Configuration for video processing"""
    language: str = "Hebrew"
    frame_interval: int = 10
    detect_scenes: bool = True
    scene_threshold: float = 30.0
    enable_ocr: bool = True
    start_time: float = 0.0
    end_time: float = 0.0
    mission: str = "general"
    generate_flowchart: bool = False
    max_batch_size_mb: float = 10.0
    max_images_per_batch: int = 15
    batch_overlap_frames: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {}
    
    # Model configuration
    config["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    if not config["api_key"]:
        config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not config["api_key"]:
        config["api_key"] = os.environ.get("HF_TOKEN", "")
    
    # Ollama configuration
    config["ollama_enabled"] = os.environ.get("OLLAMA_ENABLED", "").lower() == "true"
    config["ollama_base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    config["ollama_model"] = os.environ.get("OLLAMA_MODEL", "llama3")
    config["ollama_vision_model"] = os.environ.get("OLLAMA_VISION_MODEL", "llava")
    
    return config

def create_default_config(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Create a default configuration"""
    config = {
        "model": ModelConfig(),
        "video": VideoConfig(),
    }
    
    if api_key:
        config["model"].api_key = api_key
    
    # Load from environment if available
    env_config = load_config_from_env()
    if env_config.get("api_key") and not config["model"].api_key:
        config["model"].api_key = env_config["api_key"]
    
    # Configure Ollama if enabled in environment
    if env_config.get("ollama_enabled"):
        config["model"].model_type = "ollama"
        config["model"].ollama.enabled = True
        config["model"].ollama.base_url = env_config.get("ollama_base_url", "http://localhost:11434")
        config["model"].ollama.model_name = env_config.get("ollama_model", "llama3")
        config["model"].ollama.vision_model = env_config.get("ollama_vision_model", "llava")
    
    return config
