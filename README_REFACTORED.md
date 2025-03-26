# SmolaVision (Refactored)

This is the refactored version of SmolaVision, a system for analyzing videos using AI vision models.

## Project Structure

The refactored project follows a more modular and maintainable structure:

```
smolavision/
├── smolavision/            # Main package
│   ├── __init__.py         # Package exports
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── logging.py          # Logging configuration
│   ├── exceptions.py       # Custom exceptions
│   ├── pipeline.py         # Analysis pipeline
│   ├── video/              # Video processing module
│   ├── ocr/                # OCR module
│   ├── analysis/           # Analysis module
│   ├── models/             # Model interfaces
│   ├── batch/              # Batch processing
│   ├── tools/              # Tool implementations
│   └── utils/              # Utility functions
├── scripts/                # Command-line scripts
├── tests/                  # Test directory
├── docs/                   # Documentation
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Main documentation
```

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smolavision.git
cd smolavision

# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[all]"
```

## Running the Refactored Version

During the transition period, you can run the refactored version using:

```bash
python scripts/run_local_refactored.py path/to/video.mp4
```

Once the refactoring is complete, you can use:

```bash
smolavision --video path/to/video.mp4 --ollama-enabled
```

## Features

- Extract frames from videos at regular intervals
- Detect scene changes automatically
- Extract text from frames using OCR
- Analyze video content using AI vision models
- Generate coherent summaries of video content
- Support for workflow analysis and flowchart generation
- Support for local models via Ollama

## Using with Local Models (NVIDIA 3060 12GB)

SmolaVision supports running with local models via Ollama. For a 3060 GPU with 12GB VRAM, we've included several optimized model configurations:

| Model Type | Model Name | Parameters | Description |
|------------|------------|------------|-------------|
| Text       | phi3:mini  | 3.8B       | Microsoft's Phi-3 Mini model, good for text generation |
| Vision     | bakllava:7b| 7B         | Vision-language model based on LLaVA architecture |
| Chat       | mistral:7b | 7B         | Mistral's 7B model, good for chat and text generation |
| Fast       | gemma:2b   | 2B         | Google's Gemma 2B model, very fast |
| Tiny       | tinyllama:1.1b | 1.1B   | Extremely small model for basic tasks |

## Architecture

For more details on the architecture, see the [Architecture Documentation](docs/architecture.md).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
