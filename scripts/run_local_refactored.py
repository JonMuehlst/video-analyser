#!/usr/bin/env python
"""
Run SmolaVision with local Ollama models using the refactored codebase
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolavision.logging import configure_logging
from smolavision.config import create_default_config
from smolavision.pipeline import run_smolavision
from smolavision.utils.validation import validate_video_path
from smolavision.models.factory import ModelFactory


def main():
    """Run SmolaVision with local Ollama models"""
    # Configure logging
    logger = configure_logging(level=logging.INFO, app_name="SmolaVision-Local")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SmolaVision with local Ollama models")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--language", default="Hebrew", help="Language of text in the video")
    parser.add_argument("--frame-interval", type=int, default=5, help="Extract a frame every N seconds")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration to process in seconds (0 for entire video)")
    parser.add_argument("--start-time", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--text-model", default="phi3:mini",
                        help="Ollama model for text generation (smaller = faster)")
    parser.add_argument("--vision-model", default="llava",
                        help="Ollama model for vision tasks")
    parser.add_argument("--mission", default="general", choices=["general", "workflow"],
                        help="Analysis mission type")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR text extraction")
    parser.add_argument("--generate-flowchart", action="store_true", help="Generate workflow flowchart")
    
    args = parser.parse_args()
    
    # Create configuration with Ollama enabled
    config = create_default_config()
    
    # Configure for Ollama
    model_config = config["model"]
    model_config["model_type"] = "ollama"
    model_config["ollama"]["enabled"] = True
    model_config["ollama"]["base_url"] = "http://localhost:11434"
    model_config["ollama"]["model_name"] = args.text_model
    model_config["ollama"]["vision_model"] = args.vision_model
    model_config["api_key"] = None  # Ensure we don't try to use any API keys
    
    # Configure video processing
    video_config = config["video"]
    video_config["language"] = args.language
    video_config["frame_interval"] = args.frame_interval
    video_config["detect_scenes"] = True
    video_config["scene_threshold"] = 20.0
    video_config["enable_ocr"] = args.enable_ocr
    video_config["start_time"] = args.start_time
    video_config["end_time"] = args.start_time + args.duration if args.duration > 0 else 0
    video_config["mission"] = args.mission
    video_config["generate_flowchart"] = args.generate_flowchart
    
    # Get video path from command line or prompt user
    video_path = args.video_path
    if not video_path:
        video_path = input("Enter path to video file: ")
    
    try:
        # Validate video path
        video_path = validate_video_path(video_path)
        
        # Run SmolaVision
        logger.info(f"Starting analysis of video: {video_path}")
        logger.info(f"Using text model: {model_config['ollama']['model_name']}")
        logger.info(f"Using vision model: {model_config['ollama']['vision_model']}")
        
        if args.duration > 0:
            logger.info(f"Processing from {args.start_time}s to {args.start_time + args.duration}s")
        else:
            logger.info(f"Processing from {args.start_time}s to the end")
        
        result = run_smolavision(video_path=video_path, config=config)
        
        # Print result
        if isinstance(result, dict):
            if "error" in result:
                logger.error(f"Error: {result['error']}")
            else:
                print("\nSummary of video:")
                print("-" * 80)
                print(result["summary_text"][:1000] + "..." if len(result["summary_text"]) > 1000 else result["summary_text"])
                print("-" * 80)
                print(f"Full summary saved to: {result.get('coherent_summary', 'Not available')}")
                print(f"Full analysis saved to: {result.get('full_analysis', 'Not available')}")
                
                if "flowchart" in result:
                    print(f"Workflow flowchart saved to: {result['flowchart']}")
        else:
            logger.error(result)
    
    except Exception as e:
        logger.exception(f"Error running SmolaVision: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
