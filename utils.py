"""
Utility functions for SmolaVision
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("SmolaVision")

def ensure_directory(directory_path: str) -> str:
    """Ensure a directory exists, creating it if necessary"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file"""
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        return {}

def save_config_file(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to a JSON file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config file: {str(e)}")
        return False

def format_time_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return 0.0
