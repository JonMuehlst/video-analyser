import sys
import logging
from typing import Dict, Any, List, Optional

from smolavision.cli.parser import parse_args
# Import specific command functions
from smolavision.cli.commands import (
    run_analysis,
    check_dependencies_command,
    setup_ollama_command,
    run_anthropic_command
)
from smolavision.logging.setup import setup_logging
from smolavision.exceptions import SmolaVisionError

logger = logging.getLogger(__name__) # Use __name__ for logger

def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Determine command to run based on the subparser destination
    command = getattr(args, 'command', None)

    if command == 'check':
        return check_dependencies_command(args)
    elif command == 'setup-ollama':
        return setup_ollama_command(args)
    elif command == 'run-anthropic':
        return run_anthropic_command(args)
    elif command is None:
        # Default command is run_analysis if no subcommand is specified
        # Ensure the main parser (not a subparser) was used
        if not hasattr(args, 'video'): # Check if a required arg for run_analysis is missing
             logger.error("No command specified and required arguments for analysis are missing.")
             # Potentially print help here
             return 1
        return run_analysis(args)
    else:
        # Should not happen if parser is configured correctly
        logger.error(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    # Ensure the script can be run directly, e.g., python -m smolavision.cli.main ...
    sys.exit(main())
