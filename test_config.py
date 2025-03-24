#!/usr/bin/env python3
"""
Test script to verify that configuration is being loaded correctly from .env file.
"""

import logging
from config import create_default_config, load_config_from_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmolaVision")

def main():
    # Test loading environment configuration
    logger.info("Testing loading environment configuration...")
    env_config = load_config_from_env()
    
    # Print API key status (without revealing the actual keys)
    logger.info(f"Anthropic API key found: {bool(env_config.get('api_key'))}")
    
    # Test creating default configuration
    logger.info("Testing creating default configuration...")
    config = create_default_config()
    
    # Print model configuration
    logger.info(f"Model type: {config['model'].model_type}")
    logger.info(f"API key found: {bool(config['model'].api_key)}")
    
    # Print Ollama configuration
    logger.info(f"Ollama enabled: {config['model'].ollama.enabled}")
    if config['model'].ollama.enabled:
        logger.info(f"Ollama base URL: {config['model'].ollama.base_url}")
        logger.info(f"Ollama model: {config['model'].ollama.model_name}")
        logger.info(f"Ollama vision model: {config['model'].ollama.vision_model}")
    
    # Print video configuration
    logger.info(f"Language: {config['video'].language}")
    logger.info(f"Frame interval: {config['video'].frame_interval}")
    logger.info(f"Detect scenes: {config['video'].detect_scenes}")
    logger.info(f"Scene threshold: {config['video'].scene_threshold}")
    
    logger.info("Configuration loading test complete.")

if __name__ == "__main__":
    main()
