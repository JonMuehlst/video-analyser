#!/usr/bin/env python
"""
Setup script to pull Ollama models suitable for a 3060 GPU with 12GB RAM
"""
import os
import sys
import subprocess
import argparse
import dotenv
from pathlib import Path
from smolavision.config import create_default_config
# Import the utility function for running commands
from smolavision.utils.ollama_setup import _run_command

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Removed local run_command function

def main():
    """Pull Ollama models suitable for a 3060 GPU with 12GB RAM"""
    parser = argparse.ArgumentParser(description="Setup Ollama models for SmolaVision")
    parser.add_argument("--all", action="store_true", help="Pull all models")
    parser.add_argument("--text", action="store_true", help="Pull text model")
    parser.add_argument("--vision", action="store_true", help="Pull vision model")
    parser.add_argument("--chat", action="store_true", help="Pull chat model")
    parser.add_argument("--fast", action="store_true", help="Pull fast model")
    parser.add_argument("--tiny", action="store_true", help="Pull tiny model")
    
    args = parser.parse_args()
    
    # If no specific models are selected, pull all
    if not (args.all or args.text or args.vision or args.chat or args.fast or args.tiny):
        args.all = True
    
    # Get model names from config
    config = create_default_config()
    
    # Extract the small_models dictionary
    if "model" in config and "ollama" in config["model"]:
        small_models = config["model"]["ollama"].get("small_models", {})
    else:
        # Default models if config structure is different
        small_models = {
            "text": "phi3:mini",       # Phi-3 Mini (3.8B parameters)
            "vision": "bakllava:7b",   # Bakllava 7B (LLaVA architecture)
            "chat": "mistral:7b",      # Mistral 7B
            "fast": "gemma:2b",        # Gemma 2B
            "tiny": "tinyllama:1.1b"   # TinyLlama 1.1B
        }

    # Check if Ollama is installed using the utility function
    # Note: _run_command expects a list of arguments, not a single string with shell=True
    if not _run_command(["ollama", "--version"]):
        print("Ollama is not installed or not in PATH. Please install Ollama first.")
        print("Visit https://ollama.com/download for installation instructions.")
        return

    # Pull models using the utility function
    models_to_pull = []
    if args.all or args.text:
        models_to_pull.append(small_models['text'])
    if args.all or args.vision:
        models_to_pull.append(small_models['vision'])
    if args.all or args.chat:
        models_to_pull.append(small_models['chat'])
    if args.all or args.fast:
        models_to_pull.append(small_models['fast'])
    if args.all or args.tiny:
        models_to_pull.append(small_models['tiny'])

    # Remove duplicates if --all is used with specific flags
    models_to_pull = list(set(models_to_pull))

    all_successful = True
    for model in models_to_pull:
        print(f"\nPulling model: {model}")
        if not _run_command(["ollama", "pull", model]):
            all_successful = False
            print(f"Failed to pull model: {model}") # Logged by _run_command already

    if all_successful:
        print("\nSetup complete! You can now run SmolaVision with local models.")
    else:
        print("\nSetup incomplete. Some models failed to pull. Check logs above.")
    print("Use: python run_local.py [video_path]")

if __name__ == "__main__":
    main()
