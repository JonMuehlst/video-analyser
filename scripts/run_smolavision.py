#!/usr/bin/env python
"""
SmolaVision - Video Analysis using AI Vision Models

This script is the main entry point for running SmolaVision from the command line.
"""

import sys
import os

# Add the parent directory to the path so we can import smolavision
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smolavision.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
