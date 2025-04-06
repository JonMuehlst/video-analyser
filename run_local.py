#!/usr/bin/env python
"""
Run SmolaVision with local Ollama models
"""
import os
import sys
import argparse
import logging
import dotenv
from pathlib import Path
import requests
import time
import ollama

# Import from the new package structure
from smolavision.pipeline import run_smolavision
from smolavision.config import create_default_config
from smolavision.logging import setup_logging as configure_logging

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("smolavision_local.log")]
)
logger = logging.getLogger("SmolaVision")

# Use functions from the package instead of local implementations
from smolavision.utils.dependency_checker import (
    check_ollama_installed,
    check_ollama_running,
    list_ollama_models
)
from smolavision.utils.ollama_setup import setup_ollama_models

def check_and_setup_ollama(base_url="http://localhost:11434"):
    """Check Ollama installation, server status, and required models."""
    logger.info("Checking Ollama setup...")

    if not check_ollama_installed():
        logger.error("Ollama is not installed. Please install it from https://ollama.com")
        return False

    if not check_ollama_running(base_url):
        logger.error(f"Ollama server is not running or not reachable at {base_url}.")
        logger.error("Please start the Ollama server (e.g., run 'ollama serve') and try again.")
        return False

    logger.info("Ollama installed and server is running.")

    # Define preferred models (can be adjusted)
    preferred_text_models = ["phi3:mini", "llama3", "mistral:7b"]
    preferred_vision_models = ["llava", "bakllava:7b"]

    try:
        available = list_ollama_models(base_url)
        logger.info(f"Available Ollama models: {available}")

        # Determine which models to check/pull
        models_to_ensure = []
        text_model_available = False
        for model in preferred_text_models:
            if model in available:
                logger.info(f"Found preferred text model: {model}")
                text_model_available = True
                break
        if not text_model_available:
            models_to_ensure.append(preferred_text_models[0]) # Add the first preferred text model

        vision_model_available = False
        for model in preferred_vision_models:
            if model in available:
                logger.info(f"Found preferred vision model: {model}")
                vision_model_available = True
                break
        if not vision_model_available:
             models_to_ensure.append(preferred_vision_models[0]) # Add the first preferred vision model

        if models_to_ensure:
            logger.warning(f"Required models missing: {models_to_ensure}. Attempting to pull...")
            if not setup_ollama_models(models=models_to_ensure, base_url=base_url):
                logger.error("Failed to pull required Ollama models.")
                return False
            logger.info("Successfully pulled required models.")
        else:
            logger.info("Required Ollama models are available.")

        return True

    except Exception as e:
        logger.error(f"Error during Ollama model check/setup: {e}")
        return False

def get_available_model(preferred_models: list, available_models: list, default_model: str) -> str:
    """Select the best available model from a preferred list."""
    for model in preferred_models:
        if model in available_models:
            return model
    # Fallback if no preferred models are available
    if available_models:
         # Try to find *any* model of the right type (simple check)
         if "llava" in default_model.lower(): # Check if it's a vision model default
             vision_fallback = next((m for m in available_models if "llava" in m.lower()), None)
             if vision_fallback: return vision_fallback
         else: # Assume text model default
             text_fallback = next((m for m in available_models if "llava" not in m.lower()), None)
             if text_fallback: return text_fallback
         # If no type match, return first available
         return available_models[0]
    return default_model # Return default if nothing is available

def main():
    """Run SmolaVision with local Ollama models"""
    # Check Ollama setup first
    if not check_and_setup_ollama():
         sys.exit(1)

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
    parser.add_argument("--check-dependencies", action="store_true",
                        help="Check for required dependencies and exit")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")

    args = parser.parse_args()

    # Create configuration with Ollama enabled
    config = create_default_config()

    # --- Configure for Ollama ---
    config["model"]["model_type"] = "ollama"
    config["model"]["ollama"]["enabled"] = True
    config["model"]["ollama"]["base_url"] = "http://localhost:11434" # Default, can be overridden by env/args
    config["model"]["api_key"] = None # Ensure no cloud API key is used

    # --- Determine best available local models ---
    ollama_base_url = config["model"]["ollama"]["base_url"]
    available_models = list_ollama_models(ollama_base_url)

    # Preferred models (can be customized)
    preferred_text_models = ["phi3:mini", "llama3", "mistral:7b"]
    preferred_vision_models = ["llava", "bakllava:7b"]

    # Select models
    config["model"]["ollama"]["model_name"] = get_available_model(
        preferred_text_models, available_models, "phi3:mini"
    )
    config["model"]["ollama"]["vision_model"] = get_available_model(
        preferred_vision_models, available_models, "llava"
    )

    # Override with command-line args if provided
    if args.text_model:
        config["model"]["ollama"]["model_name"] = args.text_model
    if args.vision_model:
        config["model"]["ollama"]["vision_model"] = args.vision_model

    logger.info(f"Using Ollama text model: {config['model']['ollama']['model_name']}")
    logger.info(f"Using Ollama vision model: {config['model']['ollama']['vision_model']}")

    # --- Configure video processing from args ---
    config["video"]["language"] = args.language
    config["video"]["frame_interval"] = args.frame_interval
    config["video"]["detect_scenes"] = True # Defaulting to True for local runs
    config["video"]["scene_threshold"] = 20.0 # Adjusted default
    config["video"]["enable_ocr"] = args.enable_ocr
    config["video"]["start_time"] = args.start_time
    config["video"]["end_time"] = args.start_time + args.duration if args.duration > 0 else 0

    # --- Configure analysis from args ---
    config["analysis"]["mission"] = args.mission
    config["analysis"]["generate_flowchart"] = args.generate_flowchart
    # Optional: Adjust batch sizes for local models if needed
    # config["analysis"]["max_images_per_batch"] = 5

    # --- Get video path ---
    video_path = args.video_path
    if not video_path:
        video_path = input("Enter path to video file: ")
        
    # Check if the file exists
    if not os.path.exists(video_path):
        logger.error(f"Warning: Video file not found: {video_path}")
        print("Please provide a valid video path as an argument:")
        print("python run_local.py \"path/to/your/video.mp4\"")
        return
    
    # --- Run SmolaVision ---
    logger.info(f"Starting analysis of video: {video_path}")
    if config["video"]["end_time"] > 0:
        logger.info(f"Processing from {config['video']['start_time']:.1f}s to {config['video']['end_time']:.1f}s")
    else:
        logger.info(f"Processing from {config['video']['start_time']:.1f}s to the end")

    try:
        result = run_smolavision(video_path=video_path, config=config)

        # --- Print result ---
        if isinstance(result, dict):
            print("\n--- Analysis Summary ---")
            print(result.get("summary_text", "No summary generated."))
            print("------------------------")
            print(f"Results saved to directory: {result.get('output_dir', 'N/A')}")
            if result.get("summary_path"):
                print(f"Summary file: {result['summary_path']}")
            if result.get("full_analysis_path"):
                print(f"Full analysis file: {result['full_analysis_path']}")
            if result.get("flowchart_path"):
                print(f"Flowchart file: {result['flowchart_path']}")
        else:
            logger.error(f"Analysis did not return expected results: {result}")

    except Exception as e:
        logger.exception(f"An error occurred during SmolaVision execution: {e}")
        print(f"\nERROR: Analysis failed. Check logs ({logger.handlers[-1].baseFilename}) for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
