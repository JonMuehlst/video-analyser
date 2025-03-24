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
    config["video"]["language"] = "Hebrew"
    config["video"]["frame_interval"] = 10
    config["video"]["detect_scenes"] = True
    config["video"]["scene_threshold"] = 30.0
    config["video"]["enable_ocr"] = True
    config["video"]["start_time"] = 0.0
    config["video"]["end_time"] = 120.0  # Process first 2 minutes
    config["video"]["mission"] = "general"
    
    # Get video path from command line or use default
    video_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\jm\Videos\private\תביעה נגזרת מא'-ת' מלא.mp4"
    
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
