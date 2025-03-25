#!/usr/bin/env python
"""
List available Ollama models.
This script connects to the Ollama API and lists all available models.
"""

import argparse
import logging
import sys
from typing import List, Dict, Any

# Import local modules
from ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmolaVision")

def list_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List all available Ollama models"""
    client = OllamaClient(base_url=base_url)
    
    # Check if Ollama is running
    if not client._check_connection():
        logger.error("Cannot connect to Ollama API. Is Ollama running?")
        return []
    
    try:
        # Get list of models
        models = client.list_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

def get_model_details(model_name: str, base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    client = OllamaClient(base_url=base_url)
    
    try:
        # Get model info
        model_info = client.get_model_info(model_name)
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}

def main():
    """Command-line interface for listing Ollama models"""
    parser = argparse.ArgumentParser(description="List available Ollama models")
    parser.add_argument("--base-url", default="http://localhost:11434", 
                        help="Ollama API base URL (default: http://localhost:11434)")
    parser.add_argument("--model", help="Get detailed information about a specific model")
    
    args = parser.parse_args()
    
    if args.model:
        # Get details for a specific model
        model_info = get_model_details(args.model, args.base_url)
        if model_info:
            print(f"\nModel: {args.model}")
            print("-" * 50)
            for key, value in model_info.items():
                if key != "parameters":
                    print(f"{key}: {value}")
            
            # Print parameters in a more readable format
            if "parameters" in model_info:
                print("\nParameters:")
                for param_key, param_value in model_info["parameters"].items():
                    print(f"  {param_key}: {param_value}")
        else:
            print(f"Model '{args.model}' not found or error retrieving information.")
    else:
        # List all models
        models = list_models(args.base_url)
        
        if not models:
            print("No models found or Ollama is not running.")
            sys.exit(1)
        
        print("\nAvailable Ollama Models:")
        print("-" * 50)
        for model in models:
            print(f"- {model}")
        
        print("\nFor detailed information about a specific model, run:")
        print(f"python {sys.argv[0]} --model MODEL_NAME")

if __name__ == "__main__":
    main()
