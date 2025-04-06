import argparse
from typing import Dict, Any

def create_parser() -> argparse.ArgumentParser:
    """
    Create the command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description="SmolaVision: Analyze videos using AI vision models")
    
    # Input video
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--config", help="Path to configuration file")
    
    # Video processing options
    video_group = parser.add_argument_group("Video Processing")
    video_group.add_argument("--language", default="English", help="Language of text in the video")
    video_group.add_argument("--frame-interval", type=int, default=10, help="Extract a frame every N seconds")
    video_group.add_argument("--detect-scenes", action="store_true", help="Detect scene changes")
    video_group.add_argument("--scene-threshold", type=float, default=30.0, help="Threshold for scene change detection")
    video_group.add_argument("--enable-ocr", action="store_true", help="Enable OCR processing")
    video_group.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds")
    video_group.add_argument("--end-time", type=float, default=0.0, help="End time in seconds (0 for entire video)")
    video_group.add_argument("--resize-width", type=int, help="Width to resize frames to (keeps aspect ratio)")
    
    # Model options
    model_group = parser.add_argument_group("AI Models")
    model_group.add_argument("--model-type", choices=["anthropic", "openai", "huggingface", "ollama", "gemini"],
                        default="anthropic", help="Type of AI model to use")
    model_group.add_argument("--api-key", help="API key for the selected cloud model (e.g., Anthropic, OpenAI, Gemini)")
    model_group.add_argument("--vision-model", help="Vision model ID to use (overrides default for the selected type)")
    model_group.add_argument("--summary-model", help="Summary model to use")
    
    # Ollama options
    ollama_group = parser.add_argument_group("Ollama Integration")
    ollama_group.add_argument("--ollama-enabled", action="store_true", help="Enable Ollama integration")
    # --ollama-base-url is defined per-command where needed (check, setup-ollama)
    ollama_group.add_argument("--ollama-model", help="Ollama model name")
    ollama_group.add_argument("--ollama-vision-model", help="Ollama vision model name")
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument("--mission", choices=["general", "workflow"], default="general", 
                        help="Analysis mission type")
    analysis_group.add_argument("--generate-flowchart", action="store_true", help="Generate a flowchart from the analysis")
    analysis_group.add_argument("--max-batch-size-mb", type=float, default=10.0, help="Maximum batch size in MB")
    analysis_group.add_argument("--max-images-per-batch", type=int, default=15, help="Maximum images per batch")
    
    # Pipeline options
    pipeline_group = parser.add_argument_group("Pipeline")
    pipeline_group.add_argument("--pipeline-type", choices=["standard", "segmented"], default="standard", 
                          help="Pipeline type to use")
    pipeline_group.add_argument("--segment-length", type=int, default=300, 
                          help="Length of video segments in seconds (for segmented pipeline)")
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", default="output", help="Output directory")
    output_group.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Analysis command (default) - arguments are already defined above

    # Check dependencies command
    check_parser = subparsers.add_parser("check", help="Check dependencies")
    check_parser.add_argument("--ollama-base-url", help="Ollama server URL to check against (overrides config)")

    # Setup Ollama command
    setup_parser = subparsers.add_parser("setup-ollama", help="Setup Ollama models")
    setup_parser.add_argument("--models", help="Comma-separated list of models to install (e.g., llama3,llava)")
    setup_parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server URL")

    # Run Anthropic command (from run_anthropic.py)
    anthropic_parser = subparsers.add_parser("run-anthropic", help="Run analysis using Anthropic models")
    anthropic_parser.add_argument("--video", required=True, help="Path to the video file to analyze")
    anthropic_parser.add_argument("--api-key", help="Anthropic API key (will use env var ANTHROPIC_API_KEY if not provided)")
    anthropic_parser.add_argument("--output-dir", help="Directory to save output files", default="./output")
    anthropic_parser.add_argument("--language", help="Language for analysis and summary", default="Hebrew") # Keep Hebrew as default from original script? Or change to English? Let's keep for now.
    anthropic_parser.add_argument("--frame-interval", type=int, help="Interval between frames in seconds", default=10)
    anthropic_parser.add_argument("--scene-threshold", type=float, help="Threshold for scene detection", default=30.0)
    anthropic_parser.add_argument("--vision-model", help="Vision model to use", default="claude-3-opus-20240229")
    anthropic_parser.add_argument("--summary-model", help="Summary model to use", default="claude-3-5-sonnet-20240620")
    anthropic_parser.add_argument("--no-flowchart", action="store_true", help="Disable flowchart generation")
    anthropic_parser.add_argument("--enable-ocr", action="store_true", default=True, help="Enable OCR (default: True)") # Default to True as in original script
    anthropic_parser.add_argument("--detect-scenes", action="store_true", default=True, help="Enable scene detection (default: True)") # Default to True as in original script

    return parser

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args()
