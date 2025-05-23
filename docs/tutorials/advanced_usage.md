# Advanced Usage Tutorial

This tutorial covers advanced usage of SmolaVision, including using local models, customizing the pipeline, and programmatic usage.

## Using Local Models with Ollama

SmolaVision supports running with local models via Ollama, which is great for privacy, offline usage, or when you want to avoid API costs.

### Setting Up Ollama

1. Install Ollama from [ollama.com/download](https://ollama.com/download)

2. Start the Ollama server:
   ```bash
   ollama serve
   ```

3. Pull the required models:
   ```bash
   # Using SmolaVision's setup script
   smolavision setup-ollama --models llama3,llava
   
   # Or manually with Ollama CLI
   ollama pull llama3
   ollama pull llava
   ```

### Running with Ollama Models

```bash
smolavision --video path/to/your/video.mp4 --model-type ollama --ollama-enabled --ollama-model llama3 --ollama-vision-model llava
```

### Optimizing for Different Hardware

For different GPU configurations, you can use different model combinations:

#### High-end GPU (24GB+ VRAM)
```bash
smolavision --video path/to/your/video.mp4 --model-type ollama --ollama-enabled --ollama-model llama3:70b --ollama-vision-model bakllava
```

#### Mid-range GPU (12GB VRAM)
```bash
smolavision --video path/to/your/video.mp4 --model-type ollama --ollama-enabled --ollama-model llama3 --ollama-vision-model llava
```

#### Low-end GPU (8GB VRAM)
```bash
smolavision --video path/to/your/video.mp4 --model-type ollama --ollama-enabled --ollama-model phi3:mini --ollama-vision-model phi3:vision --max-images-per-batch 5
```

## Using Segmented Pipeline for Long Videos

For long videos, the segmented pipeline processes the video in chunks to manage memory usage:

```bash
smolavision --video path/to/your/video.mp4 --pipeline-type segmented --segment-length 300
```

This processes the video in 5-minute segments (300 seconds).

## Programmatic Usage

You can use SmolaVision programmatically in your Python code:

```python
from smolavision.config.loader import load_config
from smolavision.models.factory import ModelFactory
from smolavision.pipeline.factory import create_pipeline

# Load configuration
config = load_config()

# Override configuration options
config["video"]["frame_interval"] = 5
config["video"]["enable_ocr"] = True
config["model"]["model_type"] = "anthropic"
config["model"]["api_key"] = "your_api_key"

# Create models
vision_model = ModelFactory.create_vision_model(config["model"])
summary_model = ModelFactory.create_summary_model(config["model"])

# Create pipeline
pipeline = create_pipeline(config)
pipeline.vision_model = vision_model
pipeline.summary_model = summary_model

# Run pipeline
result = pipeline.run("path/to/your/video.mp4")

# Access results
summary = result["summary_text"]
analyses = result["analyses"]
output_dir = result["output_dir"]

print(f"Summary: {summary[:100]}...")
print(f"Output directory: {output_dir}")
```

## Custom Pipelines

You can create custom pipelines by extending the `Pipeline` base class:

```python
from typing import Dict, Any
from smolavision.pipeline.base import Pipeline
from smolavision.video.extractor import extract_frames
from smolavision.models.factory import ModelFactory

class CustomPipeline(Pipeline):
    """Custom pipeline that only extracts frames and analyzes key frames."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vision_model = ModelFactory.create_vision_model(config["model"])
    
    def run(self, video_path: str) -> Dict[str, Any]:
        # Extract frames
        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.config["video"].get("frame_interval", 10),
            detect_scenes=True
        )
        
        # Filter only scene change frames
        scene_frames = [frame for frame in frames if frame.scene_change]
        
        # Analyze each scene frame
        analyses = []
        for frame in scene_frames:
            analysis = self.vision_model.analyze_images(
                images=[frame.image_data],
                prompt=f"Describe what you see in this frame at {frame.timestamp} seconds."
            )
            analyses.append(analysis)
        
        return {
            "frames": scene_frames,
            "analyses": analyses,
            "summary": "\n\n".join(analyses)
        }
```

## Custom Tools

You can create custom tools by extending the `Tool` base class:

```python
from typing import Dict, Any
from smolavision.tools.base import Tool

class AudioExtractionTool(Tool):
    """Tool for extracting audio from a video."""
    
    name = "audio_extraction"
    description = "Extract audio from a video file"
    input_type = "video_path"
    output_type = "audio_path"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def use(self, video_path: str) -> str:
        import subprocess
        import os
        
        # Create output directory
        output_dir = self.config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        audio_path = os.path.join(output_dir, "audio.mp3")
        
        # Extract audio using ffmpeg
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path
        ], check=True)
        
        return audio_path
```

## Advanced Configuration

### Using Multiple Models

You can use different models for different parts of the pipeline:

```json
{
  "model": {
    "model_type": "anthropic",
    "api_key": "your_anthropic_key",
    "vision_model": "claude-3-opus-20240229",
    "summary_model": "claude-3-5-sonnet-20240620"
  }
}
```

### Custom Prompts (Programmatic)

Customizing prompts currently requires modifying the prompt generation logic (e.g., in `smolavision/analysis/vision.py` or `smolavision/analysis/summarization.py`) or using the library programmatically and passing custom prompts directly to the model methods. There isn't a direct configuration file setting for arbitrary custom prompts in the current structure.

*(The following sections describe features not currently implemented in the provided codebase)*

*Note: The following configuration options (`processing`, `notifications`, `storage`, `memory`, `quality`) are illustrative examples of potential future features or advanced configurations but are **not supported** by the current codebase structure as provided.*
