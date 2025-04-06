import logging
import subprocess
import platform
from typing import List, Optional

# Import checks from dependency_checker
from .dependency_checker import (
    check_ollama_installed,
    check_ollama_running,
    list_ollama_models
)

logger = logging.getLogger(__name__)

def _run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """Run a shell command and return success status, logging output."""
    command_str = ' '.join(command)
    logger.info(f"Running command: {command_str}")
    try:
        # Use subprocess.run with explicit encoding settings
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            encoding='utf-8',  # Use UTF-8 encoding
            errors='ignore',   # Ignore encoding errors
            check=False,       # Don't raise exception on non-zero exit
            text=True          # Ensure text mode
        )

        if result.stdout:
            logger.info(f"Command output:\n{result.stdout.strip()}")
        if result.stderr:
            # Log stderr as warning unless return code is non-zero
            if result.returncode != 0:
                logger.error(f"Command error output:\n{result.stderr.strip()}")
            else:
                logger.warning(f"Command stderr output:\n{result.stderr.strip()}")

        if result.returncode != 0:
            logger.error(f"Command '{command_str}' failed with exit code {result.returncode}")
            return False

        return True
    except FileNotFoundError:
        logger.error(f"Error: Command '{command[0]}' not found. Is it installed and in PATH?")
        return False
    except Exception as e:
        logger.error(f"Error running command '{command_str}': {str(e)}")
        return False

def setup_ollama_models(models: List[str], base_url: str = "http://localhost:11434") -> bool:
    """
    Check for required Ollama models and pull them if missing.

    Args:
        models: List of model names to ensure are installed (e.g., ["llama3", "llava"]).
        base_url: The base URL for the Ollama server.

    Returns:
        True if all specified models are available or successfully pulled, False otherwise.
    """
    logger.info(f"Setting up Ollama models: {models}")

    # 1. Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("Ollama is not installed. Please install it from https://ollama.com")
        return False

    # 2. Check if Ollama server is running
    if not check_ollama_running(base_url):
        logger.error(f"Ollama server is not running or not reachable at {base_url}.")
        start_cmd = "ollama serve"
        if platform.system() == "Windows":
            start_cmd = "Start Ollama application"
        logger.error(f"Please start the Ollama server (e.g., run '{start_cmd}') and try again.")
        return False

    # 3. List available models
    try:
        available_models = list_ollama_models(base_url)
        logger.info(f"Currently available Ollama models: {available_models}")
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return False

    # 4. Determine missing models
    missing_models = [model for model in models if model not in available_models]

    if not missing_models:
        logger.info("All required Ollama models are already available.")
        return True

    # 5. Pull missing models
    logger.info(f"Attempting to pull missing models: {missing_models}")
    all_pulled = True
    for model in missing_models:
        logger.info(f"Pulling model: {model}...")
        # Use the ollama CLI command to pull
        if not _run_command(["ollama", "pull", model]):
            logger.error(f"Failed to pull model: {model}")
            all_pulled = False
        else:
            logger.info(f"Successfully pulled model: {model}")

    if not all_pulled:
        logger.error("Failed to pull one or more required Ollama models.")
        return False

    logger.info("All required Ollama models are now set up.")
    return True
