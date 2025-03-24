#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional
from video_analysis import run_smolavision
from utils import load_config_file, save_config_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmolaVision-Anthropic")

def main():
    parser = argparse.ArgumentParser(description="Run SmolaVision with Anthropic models")
    parser.add_argument("--video_path", help="Path to the video file to analyze")
    parser.add_argument("--api-key", help="Anthropic API key (will use env var ANTHROPIC_API_KEY if not provided)")
    parser.add_argument("--output-dir", help="Directory to save output files", default="./output")
    parser.add_argument("--language", help="Language for analysis and summary", default="Hebrew")
    parser.add_argument("--frame-interval", type=int, help="Interval between frames in seconds", default=10)
    parser.add_argument("--scene-threshold", type=float, help="Threshold for scene detection", default=30.0)
    parser.add_argument("--vision-model", help="Vision model to use", default="claude-3-opus-20240229")
    parser.add_argument("--summary-model", help="Summary model to use", default="claude-3-5-sonnet-20240620")
    parser.add_argument("--no-flowchart", action="store_true", help="Disable flowchart generation")

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run SmolaVision with the specified parameters
    run_smolavision(
        video_path=args.video_path,
        api_key=api_key,
        language=args.language,
        frame_interval=args.frame_interval,
        detect_scenes=True,
        scene_threshold=args.scene_threshold,
        vision_model=args.vision_model,
        summary_model=args.summary_model,
        generate_flowchart=not args.no_flowchart,
        enable_ocr=True
    )

    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
