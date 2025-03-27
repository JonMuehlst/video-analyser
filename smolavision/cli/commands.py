import os
import sys
import logging
from typing import Dict, Any, List, Optional

from smolavision.config.loader import load_config
from smolavision.config.validation import validate_config
from smolavision.pipeline.factory import create_pipeline
from smolavision.exceptions import SmolaVisionError
from smolavision.cli.utils import print_results, print_error

logger = logging.getLogger(__name__)

def run_analysis(args) -> int:
    """
    Run video analysis with the given arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load and validate configuration
        config = load_config(config_path=args.config, args=args)
        is_valid, errors = validate_config(config)
        
        if not is_valid:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return 1
        
        # Create and run pipeline
        pipeline = create_pipeline(config)
        result = pipeline.run(args.video)
        
        # Print results
        print_results(result)
        
        return 0
        
    except SmolaVisionError as e:
        print_error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error")
        return 2

def check_dependencies(args) -> int:
    """
    Check if all dependencies are installed and configured.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from smolavision.utils.dependency_checker import check_dependencies
        
        # Check dependencies
        missing_deps = check_dependencies()
        
        if missing_deps:
            print_error("Missing dependencies:")
            for dep in missing_deps:
                print(f"  - {dep}")
            return 1
        
        print("All dependencies are installed and configured.")
        return 0
        
    except Exception as e:
        print_error(f"Error checking dependencies: {str(e)}")
        logger.exception("Error checking dependencies")
        return 2

def setup_ollama(args) -> int:
    """
    Set up Ollama models.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from smolavision.utils.ollama_setup import setup_ollama_models
        
        # Set up Ollama models
        success = setup_ollama_models(
            models=args.models.split(",") if args.models else ["llama3", "llava"],
            base_url=args.ollama_base_url
        )
        
        if not success:
            print_error("Failed to set up Ollama models.")
            return 1
        
        print("Ollama models set up successfully.")
        return 0
        
    except Exception as e:
        print_error(f"Error setting up Ollama: {str(e)}")
        logger.exception("Error setting up Ollama")
        return 2
