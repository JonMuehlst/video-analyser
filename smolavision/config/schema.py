from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class VideoConfig:
    """Video processing configuration."""
    language: str = "English"
    frame_interval: int = 10
    detect_scenes: bool = True
    scene_threshold: float = 30.0
    enable_ocr: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    resize_width: Optional[int] = None
    
@dataclass
class OllamaConfig:
    """Ollama-specific configuration."""
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    vision_model: str = "llava"
    
@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "anthropic"  # Supported: anthropic, openai, huggingface, ollama, gemini
    api_key: Optional[str] = None
    # Default vision/summary models - these are provider-specific
    # Anthropic defaults:
    # vision_model: str = "claude-3-opus-20240229"
    # summary_model: str = "claude-3-5-sonnet-20240620"
    # OpenAI defaults:
    # vision_model: str = "gpt-4-vision-preview"
    # summary_model: str = "gpt-4-turbo"
    # Gemini defaults (can often use the same model for both):
    vision_model: str = "gemini/gemini-1.5-flash-latest" # Default if no specific provider chosen/overridden
    summary_model: str = "gemini/gemini-1.5-flash-latest" # Default if no specific provider chosen/overridden
    temperature: float = 0.7
    max_tokens: int = 4096
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    # Add Gemini specific config if needed in future, for now API key is main thing
    
@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    mission: str = "general"  # general, workflow
    generate_flowchart: bool = False
    max_batch_size_mb: float = 10.0
    max_images_per_batch: int = 15
    batch_overlap_frames: int = 2
    
@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    pipeline_type: str = "standard"  # standard, segmented
    segment_length: int = 300  # 5 minutes in seconds
    
@dataclass
class OutputConfig:
    """Output configuration."""
    formats: List[str] = field(default_factory=lambda: ["text"])
    
@dataclass
class Config:
    """Main configuration container."""
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    output_dir: str = "output"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "video": {
                "language": self.video.language,
                "frame_interval": self.video.frame_interval,
                "detect_scenes": self.video.detect_scenes,
                "scene_threshold": self.video.scene_threshold,
                "enable_ocr": self.video.enable_ocr,
                "start_time": self.video.start_time,
                "end_time": self.video.end_time,
                "resize_width": self.video.resize_width,
            },
            "model": {
                "model_type": self.model.model_type,
                "api_key": self.model.api_key,
                "vision_model": self.model.vision_model,
                "summary_model": self.model.summary_model,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "ollama": {
                    "enabled": self.model.ollama.enabled,
                    "base_url": self.model.ollama.base_url,
                    "model_name": self.model.ollama.model_name,
                    "vision_model": self.model.ollama.vision_model,
                }
            },
            "analysis": {
                "mission": self.analysis.mission,
                "generate_flowchart": self.analysis.generate_flowchart,
                "max_batch_size_mb": self.analysis.max_batch_size_mb,
                "max_images_per_batch": self.analysis.max_images_per_batch,
                "batch_overlap_frames": self.analysis.batch_overlap_frames,
            },
            "pipeline": {
                "pipeline_type": self.pipeline.pipeline_type,
                "segment_length": self.pipeline.segment_length,
            },
            "output": {
                "formats": self.output.formats,
            },
            "output_dir": self.output_dir,
        }

    # Removed duplicate to_dict method
