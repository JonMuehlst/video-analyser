import os
import sys
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def print_results(result: Dict[str, Any]) -> None:
    """
    Print analysis results to the console.
    
    Args:
        result: Analysis results dictionary
    """
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(result.get("summary_text", "No summary generated."))
    print("="*80)
    
    # Print output file paths
    print("\nOutput files:")
    for key, path in result.items():
        if isinstance(path, str) and key.endswith("_path") and os.path.exists(path):
            print(f"  - {key}: {path}")
    
    print(f"\nAll output files are in: {result.get('output_dir', 'output')}")

def print_error(message: str) -> None:
    """
    Print error message to stderr.
    
    Args:
        message: Error message
    """
    print(f"\033[91mERROR: {message}\033[0m", file=sys.stderr)

def print_warning(message: str) -> None:
    """
    Print warning message to stderr.
    
    Args:
        message: Warning message
    """
    print(f"\033[93mWARNING: {message}\033[0m", file=sys.stderr)

def print_success(message: str) -> None:
    """
    Print success message to stdout.
    
    Args:
        message: Success message
    """
    print(f"\033[92m{message}\033[0m")

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (HH:MM:SS)
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_file_size_mb(file_path: str) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
    """
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, FileNotFoundError):
        return 0.0
