import sys
import logging
from typing import Dict, Any, List, Optional

from smolavision.cli.parser import parse_args
from smolavision.cli.commands import run_analysis, check_dependencies, setup_ollama
from smolavision.logging.setup import setup_logging
from smolavision.exceptions import SmolaVisionError

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Determine command to run
    if hasattr(args, 'command'):
        if args.command == 'check':
            return check_dependencies(args)
        elif args.command == 'setup-ollama':
            return setup_ollama(args)
    
    # Default command is to run analysis
    return run_analysis(args)

if __name__ == "__main__":
    sys.exit(main())
