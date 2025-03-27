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
    model_group.add_argument("--model-type", choices=["anthropic", "openai", "huggingface", "ollama"], 
                        default="anthropic", help="Type of AI model to use")
    model_group.add_argument("--api-key", help="API key for the selected model")
    model_group.add_argument("--vision-model", help="Vision model to use")
    model_group.add_argument("--summary-model", help="Summary model to use")
    
    # Ollama options
    ollama_group = parser.add_argument_group("Ollama Integration")
    ollama_group.add_argument("--ollama-enabled", action="store_true", help="Enable Ollama integration")
    ollama_group.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server URL")
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
    
    return parser

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args()
