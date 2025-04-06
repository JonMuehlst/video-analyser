import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_directory(directory_path: str) -> str:
    """Ensure a directory exists, creating it if necessary"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return 0.0
