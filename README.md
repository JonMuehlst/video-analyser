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
- Support for Google Gemini models via LiteLLM

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

4. **Set API Keys (for cloud models):**
   Set environment variables for the models you want to use:
   - `ANTHROPIC_API_KEY=your_anthropic_key`
   - `OPENAI_API_KEY=your_openai_key`
   - `GEMINI_API_KEY=your_google_ai_studio_key`
   - `HUGGINGFACE_API_KEY=your_hf_token` (Optional for some HF models)
   *(You can also provide keys via the `--api-key` argument or a config file)*

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

Use the built-in command to check and pull required Ollama models:

```bash
smolavision setup-ollama --models llama3,llava
```

Or use the example script for more options:

```bash
python examples/setup_ollama_models.py --all
```

### Run with Local Models

Use the `smolavision` command with Ollama options:

```bash
smolavision --video "path/to/your/video.mp4" --model-type ollama --ollama-enabled --ollama-model llama3 --ollama-vision-model llava
```

You can also use the example script `examples/run_local.py` which is pre-configured for smaller models:

```bash
python examples/run_local.py "path/to/your/video.mp4"
```

## Command Line Usage

Use the `smolavision` command:

```bash
smolavision --video "path/to/video.mp4" [options]
```

### Common Options

- `--video`: Path to the video file (required)
- `--config`: Path to a JSON configuration file
- `--language`: Language of text in the video (default: "Hebrew")
- `--frame-interval`: Extract a frame every N seconds (default: 10)
- `--detect-scenes`: Enable scene change detection
- `--scene-threshold`: Threshold for scene detection (default: 30.0)
- `--start-time`: Start time in seconds (default: 0)
- `--end-time`: End time in seconds (default: 0 = entire video)
- `--mission`: Analysis mission type ("general" or "workflow")
- `--generate-flowchart`: Generate a workflow flowchart
- `--ollama-enabled`: Use Ollama for local model inference (sets model-type to ollama)
- `--ollama-model`: Specify the Ollama text model (if using Ollama)
- `--ollama-vision-model`: Specify the Ollama vision model (if using Ollama)
- `--output-dir`: Directory to save results (default: output)
- `--verbose`: Enable detailed logging

Run `smolavision --help` for a full list of options.

## Memory Optimization Tips

When working with a 12GB GPU:

1. Use smaller batch sizes (--max-images-per-batch 5)
2. Process shorter video segments (--start-time and --end-time)
3. Use lower resolution frames (resize_width is automatically adjusted)
4. Use the smallest appropriate model for your task

## Output

The analysis results are saved to a timestamped subdirectory within the `output` directory (or the directory specified by `--output-dir`):
- `summary.txt`: Coherent summary of the video
- `full_analysis.txt`: Concatenated analysis text from all batches
- `analysis.json`/`.html`/`.md`/`.txt`: Full results in specified formats (default: `analysis.txt`)
- `flowchart.mmd`: Mermaid flowchart (if generated)
