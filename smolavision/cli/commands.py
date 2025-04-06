import os
import sys
import logging
from typing import Dict, Any, List, Optional

from smolavision.config.loader import load_config
from smolavision.config.validation import validate_config
import os
import sys
import logging
from typing import Dict, Any, List, Optional

from smolavision.config.loader import load_config, create_default_config
from smolavision.config.validation import validate_config
from smolavision.pipeline.factory import create_pipeline
from smolavision.pipeline.run import run_smolavision as run_smolavision_pipeline # Renamed import
from smolavision.exceptions import SmolaVisionError, ConfigurationError
from smolavision.cli.utils import print_results, print_error, print_warning, print_success
from smolavision.utils.dependency_checker import check_all_dependencies
# Import the actual setup function
from smolavision.utils.ollama_setup import setup_ollama_models

logger = logging.getLogger(__name__)

# Removed placeholder function setup_ollama_models

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

def run_anthropic_command(args) -> int:
    """
    Run video analysis specifically configured for Anthropic models.

    Args:
        args: Command line arguments specific to the run-anthropic command.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Get API key from args or environment
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print_error("Anthropic API key not provided. Set ANTHROPIC_API_KEY env var or use --api-key.")
            return 1

        # Create a specific config for Anthropic run
        config = create_default_config()
        config["model"]["model_type"] = "anthropic"
        config["model"]["api_key"] = api_key
        config["model"]["vision_model"] = args.vision_model
        config["model"]["summary_model"] = args.summary_model

        config["video"]["language"] = args.language
        config["video"]["frame_interval"] = args.frame_interval
        config["video"]["detect_scenes"] = args.detect_scenes # Use arg value
        config["video"]["scene_threshold"] = args.scene_threshold
        config["video"]["enable_ocr"] = args.enable_ocr # Use arg value

        config["analysis"]["generate_flowchart"] = not args.no_flowchart
        config["output_dir"] = args.output_dir

        # Validate the constructed config
        is_valid, errors = validate_config(config)
        if not is_valid:
            for error in errors:
                print_error(f"Configuration error: {error}")
            return 1

        # Run the pipeline using the high-level function
        logger.info(f"Running Anthropic analysis for video: {args.video}")
        result = run_smolavision_pipeline(video_path=args.video, config=config)

        # Print results
        print_results(result)
        print_success(f"Analysis complete. Results saved to {result.get('output_dir', args.output_dir)}")
        return 0

    except SmolaVisionError as e:
        print_error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error during Anthropic run")
        return 2


def check_dependencies_command(args) -> int:
    """
    Check if all dependencies are installed and configured.
    
    Args:
    Args:
        args: Command line arguments (currently unused for this command)

    Returns:
        Exit code (0 for success, non-zero for error)
    Args:
        args: Command line arguments (may include --config, --ollama-base-url)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        print("Checking SmolaVision dependencies...")

        # Load config to get the default ollama_base_url
        # Pass args=None to avoid double-parsing, rely on args passed to this function
        # Use args.config path if provided
        config = load_config(config_path=args.config if hasattr(args, 'config') else None, args=args)

        # Determine the Ollama base URL to use
        # Priority: 1. --ollama-base-url from check command, 2. Config value, 3. Default
        ollama_base_url = args.ollama_base_url if hasattr(args, 'ollama_base_url') and args.ollama_base_url else \
                          config.get("model", {}).get("ollama", {}).get("base_url", "http://localhost:11434")

        logger.info(f"Checking dependencies using Ollama base URL: {ollama_base_url}")
        issues = check_all_dependencies(ollama_base_url=ollama_base_url)

        if issues:
            print_error("Dependency issues found:")
            for name, issue in issues.items():
                print_warning(f"  - {name}: {issue}")
            print_error("Please resolve the issues above and try again.")
            return 1

        print_success("All dependencies seem to be installed and configured correctly.")
        return 0
        
    except Exception as e:
        print_error(f"Error checking dependencies: {str(e)}")
        logger.exception("Error checking dependencies")
        return 2

def setup_ollama_command(args) -> int:
    """
    Set up Ollama models based on command line arguments.
    
    Args:
    Args:
        args: Command line arguments specific to the setup-ollama command.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        models_to_install = args.models.split(",") if args.models else ["llama3", "llava"]
        print(f"Attempting to set up Ollama models: {models_to_install} using base URL: {args.ollama_base_url}")

        # Call the actual setup function from smolavision.utils.ollama_setup
        success = setup_ollama_models(
            models=models_to_install,
            base_url=args.ollama_base_url
        )

        if not success:
            print_error("Failed to set up one or more Ollama models. Check logs for details.")
            return 1

        print_success("Ollama models setup process completed.")
        return 0
        
    except Exception as e:
        print_error(f"Error setting up Ollama: {str(e)}")
        logger.exception("Error setting up Ollama")
        return 2
