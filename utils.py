"""
Utility functions for SmolaVision (to be deleted after refactoring)
"""
# This file is being refactored. Functions are moved to:
# - smolavision/utils/file_utils.py
# - smolavision/utils/formatting.py
# - smolavision/config/loader.py (duplicates removed)

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("SmolaVision")

# Moved to smolavision/utils/file_utils.py
# def ensure_directory(directory_path: str) -> str:
#     """Ensure a directory exists, creating it if necessary"""
#     path = Path(directory_path)
#     path.mkdir(parents=True, exist_ok=True)
#     return str(path.absolute())

# Moved to smolavision/utils/formatting.py
# def format_time_seconds(seconds: float) -> str:
#     """Format seconds as HH:MM:SS"""
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = int(seconds % 60)
#     return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Moved to smolavision/utils/file_utils.py
# def get_file_size_mb(file_path: str) -> float:
#     """Get file size in megabytes"""
#     try:
#         size_bytes = os.path.getsize(file_path)
#         return size_bytes / (1024 * 1024)
#     except Exception as e:
#         logger.error(f"Error getting file size: {str(e)}")
#         return 0.0

# Duplicates/Superseded by smolavision/config/loader.py
# def load_config_file(config_path: str) -> Dict[str, Any]: ...
# def save_config_file(config: Dict[str, Any], config_path: str) -> bool: ...
