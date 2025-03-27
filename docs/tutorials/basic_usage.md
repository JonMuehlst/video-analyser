# Basic Usage Tutorial

This tutorial will guide you through the basic usage of SmolaVision for analyzing videos.

## Installation

First, install SmolaVision and its dependencies:

```bash
pip install smolavision
```

## Analyzing a Video

The simplest way to analyze a video is using the command line interface:

```bash
smolavision --video path/to/your/video.mp4
```

This will:
1. Extract frames from the video at 10-second intervals
2. Analyze the frames using the default AI model (Anthropic Claude)
3. Generate a summary of the video content
4. Save the results to the `output` directory

## Configuration Options

You can customize the analysis with various command line options:

### Video Processing Options

```bash
# Extract frames every 5 seconds
smolavision --video path/to/your/video.mp4 --frame-interval 5

# Enable scene change detection
smolavision --video path/to/your/video.mp4 --detect-scenes

# Adjust scene change sensitivity (higher = less sensitive)
smolavision --video path/to/your/video.mp4 --detect-scenes --scene-threshold 20

# Process only a portion of the video (from 60s to 120s)
smolavision --video path/to/your/video.mp4 --start-time 60 --end-time 120

# Enable OCR to extract text from frames
smolavision --video path/to/your/video.mp4 --enable-ocr

# Specify the language of text in the video
smolavision --video path/to/your/video.mp4 --enable-ocr --language Spanish
```

### Model Options

```bash
# Use OpenAI GPT-4 instead of Claude
smolavision --video path/to/your/video.mp4 --model-type openai --api-key your_openai_key

# Use a specific vision model
smolavision --video path/to/your/video.mp4 --vision-model gpt-4o

# Use a specific summary model
smolavision --video path/to/your/video.mp4 --summary-model gpt-4-turbo
```

### Analysis Options

```bash
# Analyze workflow instead of general content
smolavision --video path/to/your/video.mp4 --mission workflow

# Generate a flowchart for workflow analysis
smolavision --video path/to/your/video.mp4 --mission workflow --generate-flowchart

# Adjust batch size for analysis
smolavision --video path/to/your/video.mp4 --max-batch-size-mb 5 --max-images-per-batch 10
```

### Output Options

```bash
# Specify output directory
smolavision --video path/to/your/video.mp4 --output-dir my_analysis

# Enable verbose logging
smolavision --video path/to/your/video.mp4 --verbose
```

## Using a Configuration File

For more complex configurations, you can use a JSON configuration file:

```json
{
  "video": {
    "language": "English",
    "frame_interval": 5,
    "detect_scenes": true,
    "scene_threshold": 20.0,
    "enable_ocr": true,
    "start_time": 60.0,
    "end_time": 120.0
  },
  "model": {
    "model_type": "anthropic",
    "api_key": "your_api_key",
    "vision_model": "claude-3-opus-20240229",
    "summary_model": "claude-3-5-sonnet-20240620"
  },
  "analysis": {
    "mission": "workflow",
    "generate_flowchart": true,
    "max_batch_size_mb": 5.0,
    "max_images_per_batch": 10
  },
  "output_dir": "my_analysis"
}
```

Save this as `config.json` and use it with:

```bash
smolavision --video path/to/your/video.mp4 --config config.json
```

## Using Environment Variables

You can also configure SmolaVision using environment variables:

```bash
# Set API key
export ANTHROPIC_API_KEY=your_api_key

# Enable Ollama
export OLLAMA_ENABLED=true
export OLLAMA_MODEL_NAME=llama3
export OLLAMA_VISION_MODEL=llava

# Run SmolaVision
smolavision --video path/to/your/video.mp4
```

## Viewing Results

After running SmolaVision, you'll find the following files in the output directory:

- `video_summary.txt`: A coherent summary of the video content
- `video_analysis_full.txt`: The complete detailed analysis
- `workflow_flowchart.mmd`: A Mermaid flowchart (if generated)

You can view the Mermaid flowchart using the [Mermaid Live Editor](https://mermaid.live/) or any Markdown viewer that supports Mermaid.
