#!/usr/bin/env python
"""
Run SmolaVision with local Ollama models
"""
import os
import sys
import requests
import time
import ollama
from video_analysis import run_smolavision
from config import create_default_config

def check_ollama_server(base_url="http://localhost:11434", max_retries=3):
    """Check if Ollama server is running and wait if it's starting up"""
    print("Checking Ollama server connection...")
    
    for attempt in range(max_retries):
        try:
            # Try using the ollama client library
            client = ollama.Client(host=base_url)
            # Try to list models to verify connection
            try:
                models_response = client.list()
                print("✓ Connected to Ollama server successfully")
                return True
            except Exception:
                # Fallback to direct API call if client.list() fails
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✓ Connected to Ollama server successfully")
                    return True
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Ollama server not responding at {base_url}")
            print(f"Error: {str(e)}")
        
        if attempt < max_retries - 1:
            print("Waiting for Ollama server to start (5 seconds)...")
            time.sleep(5)
    
    print("\nERROR: Could not connect to Ollama server.")
    print("Please make sure Ollama is installed and running:")
    print("1. Install Ollama from https://ollama.com/")
    print("2. Start the Ollama server")
    print("3. Run this script again\n")
    return False

def check_required_models(base_url="http://localhost:11434"):
    """Check if required models are available and suggest pulling them if not"""
    required_models = ["llava", "phi3:mini"]
    available_models = []
    
    try:
        # Use the ollama client library
        client = ollama.Client(host=base_url)
        models_response = client.list()
        
        # Handle different response formats
        if isinstance(models_response, dict) and 'models' in models_response:
            # New format
            if isinstance(models_response['models'], list):
                if models_response['models'] and isinstance(models_response['models'][0], dict):
                    # Try to extract model names based on available keys
                    if 'name' in models_response['models'][0]:
                        available_models = [m['name'] for m in models_response['models']]
                    elif 'model' in models_response['models'][0]:
                        available_models = [m['model'] for m in models_response['models']]
                    else:
                        # Just use the first key as identifier
                        first_key = next(iter(models_response['models'][0]))
                        available_models = [m.get(first_key, str(m)) for m in models_response['models']]
        elif isinstance(models_response, list):
            # Direct list format
            if models_response and isinstance(models_response[0], dict):
                if 'name' in models_response[0]:
                    available_models = [m['name'] for m in models_response]
                elif 'model' in models_response[0]:
                    available_models = [m['model'] for m in models_response]
                else:
                    # Just use the first key as identifier
                    first_key = next(iter(models_response[0]))
                    available_models = [m.get(first_key, str(m)) for m in models_response]
    except Exception as e:
        print(f"Could not check available models: {str(e)}")
        return False
    
    missing_models = [model for model in required_models if model not in available_models]
    
    if missing_models:
        print("\nSome required models are not available locally:")
        for model in missing_models:
            print(f"  - {model}")
        
        print("\nYou can pull these models with the following commands:")
        for model in missing_models:
            print(f"  ollama pull {model}")
        print()
        
        return False
    
    return True

def main():
    """Run SmolaVision with local Ollama models"""
    # Check if Ollama server is running
    if not check_ollama_server():
        return
    
    # Create configuration with Ollama enabled
    config = create_default_config()
    
    # Configure for Ollama with smaller models suitable for 12GB VRAM (3060)
    model_config = config["model"]
    model_config.model_type = "ollama"  # This tells the system to use Ollama
    model_config.ollama.enabled = True
    model_config.ollama.base_url = "http://localhost:11434"
    model_config.api_key = None  # Ensure we don't try to use any API keys
    
    # Add small models configuration if not present
    if not hasattr(model_config.ollama, "small_models"):
        model_config.ollama.small_models = {
            "text": "phi3:mini",
            "vision": "llava",
            "fast": "phi3:mini"
        }
    
    # Make sure we don't try to use any cloud APIs
    # First remove any existing keys
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("HF_TOKEN", None)
    
    # Set empty environment variables to prevent any API calls
    # This ensures libraries don't try to use default credentials
    os.environ["ANTHROPIC_API_KEY"] = "none"
    os.environ["OPENAI_API_KEY"] = "none"
    os.environ["HF_TOKEN"] = "none"
    
    # Disable any potential auto-authentication
    os.environ["HUGGINGFACE_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Check available models and set appropriate defaults
    try:
        # Use the ollama client library to get available models
        client = ollama.Client(host=model_config.ollama.base_url)
        models_response = client.list()
        
        # Handle different response formats
        available_models = []
        if isinstance(models_response, dict) and 'models' in models_response:
            # New format
            if isinstance(models_response['models'], list):
                if models_response['models'] and isinstance(models_response['models'][0], dict):
                    # Try to extract model names based on available keys
                    if 'name' in models_response['models'][0]:
                        available_models = [m['name'] for m in models_response['models']]
                    elif 'model' in models_response['models'][0]:
                        available_models = [m['model'] for m in models_response['models']]
                    else:
                        # Just use the first key as identifier
                        first_key = next(iter(models_response['models'][0]))
                        available_models = [m.get(first_key, str(m)) for m in models_response['models']]
        elif isinstance(models_response, list):
            # Direct list format
            if models_response and isinstance(models_response[0], dict):
                if 'name' in models_response[0]:
                    available_models = [m['name'] for m in models_response]
                elif 'model' in models_response[0]:
                    available_models = [m['model'] for m in models_response]
                else:
                    # Just use the first key as identifier
                    first_key = next(iter(models_response[0]))
                    available_models = [m.get(first_key, str(m)) for m in models_response]
        
        # Set text model based on availability
        if "phi3:mini" in available_models:
            model_config.ollama.model_name = "phi3:mini"
        elif "llama3" in available_models:
            model_config.ollama.model_name = "llama3"
        else:
            # Use the first available model as fallback
            if available_models:
                model_config.ollama.model_name = available_models[0]
            else:
                model_config.ollama.model_name = "phi3:mini"  # Default, will prompt to pull
        
        # Set vision model based on availability
        if "llava" in available_models:
            model_config.ollama.vision_model = "llava"
        elif "bakllava:7b" in available_models:
            model_config.ollama.vision_model = "bakllava:7b"
        else:
            # Use the first available vision model as fallback
            vision_models = [m for m in available_models if "llava" in m.lower()]
            if vision_models:
                model_config.ollama.vision_model = vision_models[0]
            else:
                model_config.ollama.vision_model = "llava"  # Default, will prompt to pull
    
    except Exception as e:
        print(f"Warning: Could not check available models: {str(e)}")
        # Use defaults from small_models
        model_config.ollama.model_name = model_config.ollama.small_models["text"]
        model_config.ollama.vision_model = model_config.ollama.small_models["vision"]
    
    print(f"Using text model: {model_config.ollama.model_name}")
    print(f"Using vision model: {model_config.ollama.vision_model}")
    
    # Check if required models are available
    check_required_models()
    
    # Configure video processing
    video_config = config["video"]
    video_config.language = "English"  # Default to English for broader compatibility
    video_config.frame_interval = 10
    video_config.detect_scenes = True
    video_config.scene_threshold = 30.0
    video_config.enable_ocr = False  # Disable OCR by default as it requires additional dependencies
    video_config.start_time = 0.0
    video_config.end_time = 120.0  # Process first 2 minutes
    video_config.mission = "general"
    
    # Optimize batch size for local processing
    video_config.max_batch_size_mb = 5.0  # Smaller batches to prevent OOM errors
    video_config.max_images_per_batch = 5  # Fewer images per batch for local processing
    
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
    
    try:
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
    except Exception as e:
        print(f"Error running SmolaVision: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
