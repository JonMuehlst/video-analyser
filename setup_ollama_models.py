#!/usr/bin/env python
"""
Setup script to pull Ollama models suitable for a 3060 GPU with 12GB RAM
"""
import os
import sys
import subprocess
import argparse
from config import create_default_config

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    try:
        # Use subprocess.run with explicit encoding settings
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            encoding='utf-8',  # Use UTF-8 encoding
            errors='ignore',   # Ignore encoding errors
            check=False        # Don't raise exception on non-zero exit
        )
        
        if result.stdout:
            print(result.stdout)
            
        if result.stderr:
            print(f"Error: {result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"Command execution error: {str(e)}")
        return False

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
    small_models = config["model"].ollama.small_models
    
    # Convert to regular dictionary if it's not already
    if not isinstance(small_models, dict):
        small_models = {
            "text": "phi3:mini",       # Phi-3 Mini (3.8B parameters)
            "vision": "bakllava:7b",   # Bakllava 7B (LLaVA architecture)
            "chat": "mistral:7b",      # Mistral 7B
            "fast": "gemma:2b",        # Gemma 2B
            "tiny": "tinyllama:1.1b"   # TinyLlama 1.1B
        }
    
    # Check if Ollama is installed
    if not run_command("ollama --version"):
        print("Ollama is not installed or not in PATH. Please install Ollama first.")
        print("Visit https://ollama.com/download for installation instructions.")
        return
    
    # Pull models
    if args.all or args.text:
        print(f"\nPulling text model: {small_models['text']}")
        run_command(f"ollama pull {small_models['text']}")
    
    if args.all or args.vision:
        print(f"\nPulling vision model: {small_models['vision']}")
        run_command(f"ollama pull {small_models['vision']}")
    
    if args.all or args.chat:
        print(f"\nPulling chat model: {small_models['chat']}")
        run_command(f"ollama pull {small_models['chat']}")
    
    if args.all or args.fast:
        print(f"\nPulling fast model: {small_models['fast']}")
        run_command(f"ollama pull {small_models['fast']}")
    
    if args.all or args.tiny:
        print(f"\nPulling tiny model: {small_models['tiny']}")
        run_command(f"ollama pull {small_models['tiny']}")
    
    print("\nSetup complete! You can now run SmolaVision with local models.")
    print("Use: python run_local.py [video_path]")

if __name__ == "__main__":
    main()
