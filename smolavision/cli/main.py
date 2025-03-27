import argparse
import logging
import sys
from typing import Dict, Any, List, Optional

from smolavision.config.loader import load_config
from smolavision.config.validation import validate_config
from smolavision.tools.factory import ToolFactory
from smolavision.logging.setup import setup_logging
from smolavision.exceptions import SmolaVisionError

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SmolaVision: Analyze videos using AI")

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--language", default="English", help="Language of text in the video")
    parser.add_argument("--frame-interval", type=int, default=10, help="Extract a frame every N seconds")
    parser.add_argument("--detect-scenes", action="store_true", help="Detect scene changes")
    parser.add_argument("--scene-threshold", type=float, default=30.0, help="Threshold for scene change detection")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR processing")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, default=0.0, help="End time in seconds (0 for entire video)")
    parser.add_argument("--resize-width", type=int, help="Width to resize frames to (keeps aspect ratio)")
    parser.add_argument("--model-type", choices=["anthropic", "openai", "huggingface", "ollama"], 
                        default="anthropic", help="Type of AI model to use")
    parser.add_argument("--api-key", help="API key for the selected model")
    parser.add_argument("--vision-model", help="Vision model to use")
    parser.add_argument("--summary-model", help="Summary model to use")
    parser.add_argument("--mission", choices=["general", "workflow"], default="general", 
                        help="Analysis mission type")
    parser.add_argument("--generate-flowchart", action="store_true", help="Generate a flowchart from the analysis")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--ollama-enabled", action="store_true", help="Enable Ollama integration")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--ollama-model", help="Ollama model name")
    parser.add_argument("--ollama-vision-model", help="Ollama vision model name")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    # Set up logging
    setup_logging()
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Load and validate configuration
        config = load_config(args.config, args)
        is_valid, errors = validate_config(config)
        
        if not is_valid:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return 1
        
        # Create tools
        frame_extraction_tool = ToolFactory.create_tool(config, "frame_extraction")
        
        # Extract frames
        logger.info(f"Extracting frames from {args.video}")
        frames_str = frame_extraction_tool.use(args.video)
        frames = eval(frames_str)  # Convert string representation back to list
        
        # Extract text with OCR if enabled
        if config["video"].get("enable_ocr", False):
            logger.info("Extracting text with OCR")
            ocr_tool = ToolFactory.create_tool(config, "ocr_extraction")
            frames_str = ocr_tool.use(frames)
            frames = eval(frames_str)
        
        # Create batches
        logger.info("Creating batches")
        batch_tool = ToolFactory.create_tool(config, "batch_creation")
        batches_str = batch_tool.use(frames)
        batches = eval(batches_str)
        
        # Analyze batches
        logger.info(f"Analyzing {len(batches)} batches")
        vision_tool = ToolFactory.create_tool(config, "vision_analysis")
        analyses = []
        previous_context = ""
        
        for i, batch in enumerate(batches):
            logger.info(f"Analyzing batch {i+1}/{len(batches)}")
            prompt = f"Analyze these frames from a video. {config['analysis'].get('mission', 'general')} analysis."
            analysis = vision_tool.use([batch], prompt, previous_context)
            analyses.append(analysis)
            previous_context = analysis[-1000:] if len(analysis) > 1000 else analysis
        
        # Generate summary
        logger.info("Generating summary")
        summary_tool = ToolFactory.create_tool(config, "summarization")
        summary = summary_tool.use(analyses, config["video"].get("language", "English"))
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
        return 0
        
    except SmolaVisionError as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 2

if __name__ == "__main__":
    sys.exit(main())
