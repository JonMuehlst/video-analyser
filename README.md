# SmolaVision

A system for analyzing videos using smolagents and vision models.

## Features

- Extract frames from videos at regular intervals
- Detect scene changes automatically
- Extract text from frames using OCR
- Analyze video content using AI vision models
- Generate coherent summaries of video content
- Support for workflow analysis and flowchart generation
- Support for local models via Ollama

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/summarize-video.git
   cd summarize-video
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Ollama (for local models):
   Visit [ollama.com/download](https://ollama.com/download) and follow the instructions for your platform.

## Using with Local Models (NVIDIA 3060 12GB)

SmolaVision supports running with local models via Ollama. For a 3060 GPU with 12GB VRAM, we've included several optimized model configurations:

| Model Type | Model Name | Parameters | Description |
|------------|------------|------------|-------------|
| Text       | phi3:mini  | 3.8B       | Microsoft's Phi-3 Mini model, good for text generation |
| Vision     | bakllava:7b| 7B         | Vision-language model based on LLaVA architecture |
| Chat       | mistral:7b | 7B         | Mistral's 7B model, good for chat and text generation |
| Fast       | gemma:2b   | 2B         | Google's Gemma 2B model, very fast |
| Tiny       | tinyllama:1.1b | 1.1B   | Extremely small model for basic tasks |

### Setup Local Models

Run the setup script to download the models:

```
python setup_ollama_models.py --all
```

Or download specific models:

```
python setup_ollama_models.py --vision --text
```

### Run with Local Models

Use the `run_local.py` script to analyze a video with local models:

```
python run_local.py "path/to/your/video.mp4"
```

This will use the smaller models configured for your GPU.

## Command Line Usage

For more control, use the main script with specific options:

```
python video_analysis.py --video "path/to/video.mp4" --ollama-enabled --start-time 0 --end-time 120
```

### Options

- `--video`: Path to the video file (required)
- `--language`: Language of text in the video (default: "Hebrew")
- `--frame-interval`: Extract a frame every N seconds (default: 10)
- `--detect-scenes`: Enable scene change detection
- `--scene-threshold`: Threshold for scene detection (default: 30.0)
- `--start-time`: Start time in seconds (default: 0)
- `--end-time`: End time in seconds (default: 0 = entire video)
- `--mission`: Analysis mission type ("general" or "workflow")
- `--generate-flowchart`: Generate a workflow flowchart
- `--ollama-enabled`: Use Ollama for local model inference
- `--ollama-model`: Specify the Ollama text model
- `--ollama-vision-model`: Specify the Ollama vision model

## Memory Optimization Tips

When working with a 12GB GPU:

1. Use smaller batch sizes (--max-images-per-batch 5)
2. Process shorter video segments (--start-time and --end-time)
3. Use lower resolution frames (resize_width is automatically adjusted)
4. Use the smallest appropriate model for your task

## Output

The analysis results are saved to the `output` directory:
- `video_analysis_full.txt`: Complete detailed analysis
- `video_summary.txt`: Coherent summary of the video
- `workflow_flowchart.mmd`: Mermaid flowchart (if generated)
