#!/usr/bin/env python
"""
Run SmolaVision with local Ollama models
"""
import os
import sys
import argparse
import logging
from video_analysis import run_smolavision
from config import create_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("smolavision_local.log")]
)
logger = logging.getLogger("SmolaVision")

def check_ollama_models(ollama_config):
    """Check if required Ollama models are available, pull if not"""
    try:
        from ollama_client import OllamaClient
        
        client = OllamaClient(base_url=ollama_config.base_url)
        
        # Check connection to Ollama
        if not client._check_connection():
            logger.error(f"Cannot connect to Ollama at {ollama_config.base_url}")
            logger.error("Please make sure Ollama is running")
            return False
            
        # Get available models
        available_models = client.list_models()
        logger.info(f"Available Ollama models: {', '.join(available_models)}")
        
        # Check if required models are available
        required_models = [
            ollama_config.model_name,
            ollama_config.vision_model
        ]
        
        missing_models = [model for model in required_models if model not in available_models]
        
        if missing_models:
            logger.warning(f"Missing required models: {', '.join(missing_models)}")
            logger.warning("Please install them with:")
            for model in missing_models:
                logger.warning(f"  ollama pull {model}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking Ollama models: {str(e)}")
        return False

def main():
    """Run SmolaVision with local Ollama models"""
    parser = argparse.ArgumentParser(description="Run SmolaVision with local Ollama models")
    
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--language", default="Hebrew", help="Language of text in the video")
    parser.add_argument("--frame-interval", type=int, default=5, help="Extract a frame every N seconds")
    parser.add_argument("--duration", type=float, default=0, 
                        help="Duration to process in seconds (0 for entire video)")
    parser.add_argument("--start-time", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--text-model", default="phi3:mini", 
                        help="Ollama model for text generation (smaller = faster)")
    parser.add_argument("--vision-model", default="bakllava:7b", 
                        help="Ollama model for vision tasks")
    parser.add_argument("--mission", default="general", choices=["general", "workflow"],
                        help="Analysis mission type")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR text extraction")
    parser.add_argument("--generate-flowchart", action="store_true", help="Generate workflow flowchart")
    
    args = parser.parse_args()
    
    # Create configuration with Ollama enabled
    config = create_default_config()
    
    # Configure for Ollama with smaller models suitable for consumer GPUs
    model_config = config["model"]
    model_config.model_type = "ollama"
    model_config.ollama.enabled = True
    model_config.ollama.base_url = "http://localhost:11434"
    model_config.ollama.model_name = args.text_model
    model_config.ollama.vision_model = args.vision_model
    
    # Configure video processing
    video_config = config["video"]
    video_config.language = args.language
    video_config.frame_interval = args.frame_interval
    video_config.detect_scenes = True
    video_config.scene_threshold = 20.0
    video_config.enable_ocr = args.enable_ocr
    video_config.start_time = args.start_time
    video_config.end_time = args.start_time + args.duration if args.duration > 0 else 0
    video_config.mission = args.mission
    video_config.generate_flowchart = args.generate_flowchart
    
    # Check if Ollama is running and has required models
    if not check_ollama_models(model_config.ollama):
        logger.error("Ollama setup incomplete. Please install required models and try again.")
        return
    
    # Get video path from command line or prompt user
    video_path = args.video_path
    if not video_path:
        video_path = input("Enter path to video file: ")
        
    # Check if the file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Run SmolaVision
    logger.info(f"Starting analysis of video: {video_path}")
    logger.info(f"Using text model: {model_config.ollama.model_name}")
    logger.info(f"Using vision model: {model_config.ollama.vision_model}")
    
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
            print(f"Full summary saved to: {result['coherent_summary']}")
            print(f"Full analysis saved to: {result['full_analysis']}")
            
            if "flowchart" in result:
                print(f"Workflow flowchart saved to: {result['flowchart']}")
    else:
        logger.error(result)

if __name__ == "__main__":
    main()
