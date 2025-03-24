"""
Configuration module for SmolaVision
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

@dataclass
class OllamaConfig:
    """Configuration for Ollama models"""
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    vision_model: str = "llava"
    # Models suitable for 12GB VRAM (3060)
    small_models: Dict[str, str] = field(default_factory=lambda: {
        "text": "phi3:mini",       # Phi-3 Mini (3.8B parameters)
        "vision": "bakllava:7b",   # Bakllava 7B (LLaVA architecture)
        "chat": "mistral:7b",      # Mistral 7B
        "fast": "gemma:2b",        # Gemma 2B
        "tiny": "tinyllama:1.1b"   # TinyLlama 1.1B
    })
    
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
    frame_interval: int = 5        # Reduced interval for more frames
    detect_scenes: bool = True     # Enable scene detection by default
    scene_threshold: float = 20.0  # Lower threshold for more sensitive detection
    enable_ocr: bool = True
    start_time: float = 0.0
    end_time: float = 0.0
    mission: str = "general"
    generate_flowchart: bool = False
    max_batch_size_mb: float = 10.0
    max_images_per_batch: int = 15
    batch_overlap_frames: int = 2
    min_scene_duration: float = 1.0  # Minimum duration between scenes in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    # Load .env file if it exists
    load_dotenv()
    
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
    
    # Video configuration
    config["language"] = os.environ.get("DEFAULT_LANGUAGE", "Hebrew")
    config["frame_interval"] = int(os.environ.get("FRAME_INTERVAL", "5"))
    config["detect_scenes"] = os.environ.get("DETECT_SCENES", "true").lower() == "true"  # Default to true
    config["scene_threshold"] = float(os.environ.get("SCENE_THRESHOLD", "20.0"))
    config["min_scene_duration"] = float(os.environ.get("MIN_SCENE_DURATION", "1.0"))
    config["enable_ocr"] = os.environ.get("ENABLE_OCR", "").lower() == "true"
    config["start_time"] = float(os.environ.get("START_TIME", "0.0"))
    config["end_time"] = float(os.environ.get("END_TIME", "0.0"))
    config["mission"] = os.environ.get("MISSION", "general")
    config["generate_flowchart"] = os.environ.get("GENERATE_FLOWCHART", "").lower() == "true"
    config["max_batch_size_mb"] = float(os.environ.get("MAX_BATCH_SIZE_MB", "10.0"))
    config["max_images_per_batch"] = int(os.environ.get("MAX_IMAGES_PER_BATCH", "15"))
    config["batch_overlap_frames"] = int(os.environ.get("BATCH_OVERLAP_FRAMES", "2"))
    
    return config

def create_default_config(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Create a default configuration"""
    # Ensure .env file is loaded
    load_dotenv()
    
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
    
    # Configure video settings from environment
    if env_config.get("language"):
        config["video"].language = env_config.get("language")
    if env_config.get("frame_interval"):
        config["video"].frame_interval = env_config.get("frame_interval")
    if "detect_scenes" in env_config:
        config["video"].detect_scenes = env_config.get("detect_scenes")
    if env_config.get("scene_threshold"):
        config["video"].scene_threshold = env_config.get("scene_threshold")
    if env_config.get("min_scene_duration"):
        config["video"].min_scene_duration = env_config.get("min_scene_duration")
    if "enable_ocr" in env_config:
        config["video"].enable_ocr = env_config.get("enable_ocr")
    if env_config.get("start_time") is not None:
        config["video"].start_time = env_config.get("start_time")
    if env_config.get("end_time") is not None:
        config["video"].end_time = env_config.get("end_time")
    if env_config.get("mission"):
        config["video"].mission = env_config.get("mission")
    if "generate_flowchart" in env_config:
        config["video"].generate_flowchart = env_config.get("generate_flowchart")
    if env_config.get("max_batch_size_mb"):
        config["video"].max_batch_size_mb = env_config.get("max_batch_size_mb")
    if env_config.get("max_images_per_batch"):
        config["video"].max_images_per_batch = env_config.get("max_images_per_batch")
    if env_config.get("batch_overlap_frames"):
        config["video"].batch_overlap_frames = env_config.get("batch_overlap_frames")
    
    return config
