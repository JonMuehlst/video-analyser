# Running SmolaVision Locally with Ollama

This guide explains how to run SmolaVision completely locally using Ollama for AI inference, without requiring any API keys or cloud services.

## Prerequisites

1. **Install Ollama**
   - Download and install from [ollama.com](https://ollama.com/)
   - Make sure the Ollama service is running

2. **Install Python Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Pull Required Models**
   ```
   ollama pull llava       # Vision model for analyzing frames
   ollama pull phi3:mini   # Text model for summarization (smaller, faster)
   ```
   
   Alternative models if you have more GPU memory:
   ```
   ollama pull llava:34b   # Better vision model (requires 24GB+ VRAM)
   ollama pull llama3      # Better text model (requires 16GB+ VRAM)
   ```

## Running the Analysis

1. **Basic Usage**
   ```
   python run_local.py "path/to/your/video.mp4"
   ```

2. **Advanced Usage**
   You can modify parameters in `run_local.py` to customize:
   - Frame interval
   - Scene detection sensitivity
   - Language
   - OCR settings
   - Processing duration

## Optional: OCR Support

To enable OCR (Optical Character Recognition) for text extraction from video frames:

1. **Install Tesseract OCR**:
   - Windows: Download and install from [github.com/UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

2. **Install Python Package**:
   - Uncomment the `pytesseract` line in requirements.txt
   - Run: `pip install pytesseract`

3. **Enable OCR in run_local.py**:
   - Set `video_config.enable_ocr = True`

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce the batch size in `run_local.py` by setting `max_images_per_batch` to a lower value
   - Use smaller models (phi3:mini instead of llama3)
   - Process shorter video segments by adjusting `start_time` and `end_time`

2. **Slow Processing**
   - Increase frame interval to process fewer frames
   - Disable scene detection
   - Disable OCR if not needed

3. **Model Not Found Errors**
   - Make sure you've pulled the required models with `ollama pull <model_name>`
   - Check available models with `ollama list`

## Hardware Requirements

- Minimum: NVIDIA GPU with 8GB VRAM
- Recommended: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- CPU-only: Possible but very slow

## Output Files

All analysis results are saved to the `output` directory:
- `video_analysis_full.txt`: Complete detailed analysis
- `video_summary.txt`: Coherent summary of the video
- `batch_analyses/`: Individual batch analysis files
