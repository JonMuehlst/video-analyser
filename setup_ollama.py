#!/usr/bin/env python
"""
Setup script for Ollama models used by SmolaVision
"""
import os
import sys
import subprocess
import argparse
import logging
import platform
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmolaVision-Setup")

def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """Run a shell command and return success status"""
    try:
        logger.info(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        if platform.system() == "Windows":
            command = ["where", "ollama"]
        else:
            command = ["which", "ollama"]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def check_ollama_running() -> bool:
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except Exception:
        return False

def install_ollama_models(models: List[str]) -> bool:
    """Install Ollama models"""
    success = True
    for model in models:
        logger.info(f"Installing model: {model}")
        if not run_command(["ollama", "pull", model]):
            logger.error(f"Failed to install model: {model}")
            success = False
    return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup Ollama models for SmolaVision")
    
    parser.add_argument("--install-all", action="store_true", help="Install all recommended models")
    parser.add_argument("--install-minimal", action="store_true", help="Install minimal set of models")
    parser.add_argument("--install-models", nargs="+", help="Install specific models")
    parser.add_argument("--list-recommended", action="store_true", help="List recommended models")
    
    args = parser.parse_args()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("Ollama is not installed. Please install it first.")
        logger.error("Visit https://ollama.com/download for installation instructions.")
        return
    
    # Check if Ollama server is running
    if not check_ollama_running():
        logger.error("Ollama server is not running. Please start it first.")
        if platform.system() == "Windows":
            logger.error("Run Ollama from the Start menu or command line.")
        else:
            logger.error("Run 'ollama serve' in a separate terminal.")
        return
    
    # Define model sets
    minimal_models = [
        "phi3:mini",      # Small text model (3.8B parameters)
        "bakllava:7b"     # Small vision model
    ]
    
    recommended_models = minimal_models + [
        "llama3",         # Better text model
        "llava",          # Better vision model
        "mistral:7b",     # Alternative text model
        "gemma:2b"        # Tiny text model for low memory
    ]
    
    # List recommended models if requested
    if args.list_recommended:
        print("Recommended models for SmolaVision:")
        print("\nMinimal set (good for 8GB VRAM):")
        for model in minimal_models:
            print(f"  - {model}")
        
        print("\nFull recommended set (for 12GB+ VRAM):")
        for model in recommended_models:
            print(f"  - {model}")
        return
    
    # Install models
    if args.install_minimal:
        logger.info("Installing minimal set of models...")
        install_ollama_models(minimal_models)
    elif args.install_all:
        logger.info("Installing all recommended models...")
        install_ollama_models(recommended_models)
    elif args.install_models:
        logger.info(f"Installing specified models: {', '.join(args.install_models)}")
        install_ollama_models(args.install_models)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
