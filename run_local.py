#!/usr/bin/env python
"""
Run SmolaVision with local Ollama models
"""
import os
import sys
from video_analysis import run_smolavision
from config import create_default_config

def main():
    """Run SmolaVision with local Ollama models"""
    # Create configuration with Ollama enabled
    config = create_default_config()
    
    # Configure for Ollama with smaller models suitable for 12GB VRAM (3060)
    model_config = config["model"]
    model_config.model_type = "ollama"
    model_config.ollama.enabled = True
    model_config.ollama.base_url = "http://localhost:11434"
    model_config.ollama.model_name = model_config.ollama.small_models["text"]  # Use phi3:mini
    model_config.ollama.vision_model = model_config.ollama.small_models["vision"]  # Use bakllava:7b
    
    # Configure video processing
    video_config = config["video"]
    video_config.language = "Hebrew"
    video_config.frame_interval = 10
    video_config.detect_scenes = True
    video_config.scene_threshold = 30.0
    video_config.enable_ocr = True
    video_config.start_time = 0.0
    video_config.end_time = 120.0  # Process first 2 minutes
    video_config.mission = "general"
    
    # Get video path from command line or use default
    # Use a safer default path without non-ASCII characters
    default_path = r"C:\Users\jm\Videos\private\video.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print(f"No video path provided, using default: {default_path}")
        video_path = default_path
        
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        print("Please provide a valid video path as an argument:")
        print("python run_local.py \"path/to/your/video.mp4\"")
        return
    
    # Run SmolaVision
    result = run_smolavision(video_path=video_path, config=config)
    
    # Print result
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSummary of video:")
            print("-" * 80)
            print(result["summary_text"][:1000] + "..." if len(result["summary_text"]) > 1000 else result["summary_text"])
            print("-" * 80)
            print(f"Full summary saved to: {result['coherent_summary']}")
            print(f"Full analysis saved to: {result['full_analysis']}")
    else:
        print(result)

if __name__ == "__main__":
    main()
