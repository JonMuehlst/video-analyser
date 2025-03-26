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
from smolavision.logging import configure_logging

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("smolavision_local.log")]
)
logger = logging.getLogger("SmolaVision")

def check_ollama_installed():
    """Check if Ollama is installed and running"""
    try:
        import subprocess

        # Check if Ollama is installed
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Ollama is not installed. Please install it from https://ollama.com")
                return False
        except Exception:
            # On Windows, try where instead of which
            try:
                result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("Ollama is not installed. Please install it from https://ollama.com")
                    return False
            except Exception:
                logger.error("Ollama is not installed. Please install it from https://ollama.com")
                return False

        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                logger.error("Ollama is installed but not running. Please start it with 'ollama serve'")
                return False
        except Exception:
            logger.error("Ollama is installed but not running. Please start it with 'ollama serve'")
            return False

        return True
    except ImportError:
        logger.error("Required package 'requests' is not installed. Please install it with 'pip install requests'")
        return False

def check_ollama_models(ollama_config):
    """Check if required Ollama models are available, pull if not"""
    try:
        # First check if Ollama is installed and running
        if not check_ollama_installed():
            return False

        from ollama_client import OllamaClient

        client = OllamaClient(base_url=ollama_config.base_url)

        # Check connection to Ollama
        if not client._check_connection():
            logger.error(f"Cannot connect to Ollama at {ollama_config.base_url}")
            logger.error("Please make sure Ollama is running with 'ollama serve'")
            return False

        # Get available models
        available_models = client.list_models()
        logger.info(f"Available Ollama models: {', '.join(available_models) if available_models else 'None'}")

        # Check if required models are available
        required_models = [
            ollama_config.model_name,
            ollama_config.vision_model
        ]

        missing_models = [model for model in required_models if model not in available_models]

        if missing_models:
            logger.warning(f"Missing required models: {', '.join(missing_models)}")
            logger.warning("Do you want to pull the missing models now? (y/n)")
            choice = input().lower()

            if choice == 'y' or choice == 'yes':
                for model in missing_models:
                    logger.info(f"Pulling model: {model}...")
                    if client.pull_model(model):
                        logger.info(f"Successfully pulled model: {model}")
                    else:
                        logger.error(f"Failed to pull model: {model}")
                        return False
            else:
                logger.warning("Please install the missing models manually with:")
                for model in missing_models:
                    logger.warning(f"  ollama pull {model}")
                return False

        return True
    except ImportError:
        logger.error("Required package 'requests' is not installed. Please install it with 'pip install requests'")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama models: {str(e)}")
        return False


def check_ollama_server(base_url="http://localhost:11434", max_retries=3):
    """Check if Ollama server is running and wait if it's starting up"""
    print("Checking Ollama server connection...")

    for attempt in range(max_retries):
        try:
            # Try direct API call first (more reliable)
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✓ Connected to Ollama server successfully")
                return True
                
            # Fallback to using the ollama client library
            client = ollama.Client(host=base_url)
            # Try to list models to verify connection
            models_response = client.list()
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
    # Define alternative models for each category
    text_models = ["phi3:mini", "llama3", "mistral:7b", "gemma:2b", "tinyllama:1.1b"]
    vision_models = ["llava", "bakllava:7b", "llava:7b", "llava:13b"]
    
    available_models = []

    try:
        # Use direct API call (more reliable)
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
        else:
            # Fallback to using the ollama client library
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

    # Check if we have at least one text model and one vision model
    has_text_model = any(model in available_models for model in text_models)
    has_vision_model = any(model in available_models for model in vision_models)
    
    missing_categories = []
    if not has_text_model:
        missing_categories.append(("text model", text_models))
    if not has_vision_model:
        missing_categories.append(("vision model", vision_models))

    if missing_categories:
        print("\nSome required model types are not available locally:")
        for category, models in missing_categories:
            print(f"  - Missing {category}. Need at least one of: {', '.join(models)}")

        print("\nYou can pull these models with the following commands:")
        for category, models in missing_categories:
            # Suggest the smallest model in each category
            suggested_model = models[0]
            print(f"  ollama pull {suggested_model}  # {category}")
        print()

        return False

    print(f"Available models: {', '.join(available_models)}")
    return True

def main():
    """Run SmolaVision with local Ollama models"""
    # Check if Ollama server is running
    if not check_ollama_server():
        return

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

    # Check for required Python packages
    try:
        import cv2
        import numpy as np
        import requests
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.error("Please install required packages with: pip install opencv-python numpy requests")
        return

    # If just checking dependencies, exit after checks
    if args.check_dependencies:
        if check_ollama_installed():
            logger.info("All dependencies are installed correctly.")
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
        # Use direct API call (more reliable)
        import requests
        response = requests.get(f"{model_config.ollama.base_url}/api/tags", timeout=5)
        
        available_models = []
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
        else:
            # Fallback to using the ollama client library
            client = ollama.Client(host=model_config.ollama.base_url)
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

        # Define model categories
        text_models = ["phi3:mini", "llama3", "mistral:7b", "gemma:2b", "tinyllama:1.1b"]
        vision_models = ["llava", "bakllava:7b", "llava:7b", "llava:13b"]
        
        # Set text model based on availability
        text_model_found = False
        for preferred_model in text_models:
            if preferred_model in available_models:
                model_config.ollama.model_name = preferred_model
                text_model_found = True
                break
                
        if not text_model_found:
            # Use the first available model as fallback
            if available_models:
                model_config.ollama.model_name = available_models[0]
                print(f"No preferred text models found. Using {available_models[0]} instead.")
            else:
                model_config.ollama.model_name = "phi3:mini"  # Default, will prompt to pull
                print("No models found. Will try to use phi3:mini (needs to be pulled).")

        # Set vision model based on availability - prioritize llava
        vision_model_found = False
        for preferred_model in vision_models:
            if preferred_model in available_models:
                model_config.ollama.vision_model = preferred_model
                vision_model_found = True
                break
                
        if not vision_model_found:
            # Use the first available vision model as fallback
            vision_models_available = [m for m in available_models if "llava" in m.lower()]
            if vision_models_available:
                model_config.ollama.vision_model = vision_models_available[0]
                print(f"Using vision model: {vision_models_available[0]}")
            else:
                model_config.ollama.vision_model = "llava"  # Default, will prompt to pull
                print("No vision models found. Will try to use llava (needs to be pulled).")

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
        logger.error("You can run this script with --check-dependencies to verify your setup.")
        return
    
    # Get video path from command line or prompt user
    video_path = args.video_path
    if not video_path:
        video_path = input("Enter path to video file: ")
        
    # Check if the file exists
    if not os.path.exists(video_path):
        logger.error(f"Warning: Video file not found: {video_path}")
        print("Please provide a valid video path as an argument:")
        print("python run_local.py \"path/to/your/video.mp4\"")
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
