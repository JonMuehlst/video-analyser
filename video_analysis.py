#!/usr/bin/env python
"""
SmolaVision: A system for analyzing videos using smolagents and vision models.
Process a video by extracting frames, sending to vision AI models, and creating a cohesive summary.
"""

import os
import time
import argparse
import logging
from typing import List, Dict, Any, Optional

# Import smolagents
from smolagents import CodeAgent, LiteLLMModel, HfApiModel

# Import local modules
from config import create_default_config
from tools import (
    FrameExtractionTool,
    OCRExtractionTool,
    BatchCreationTool,
    VisionAnalysisTool,
    SummarizationTool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("smolavision.log")]
)
logger = logging.getLogger("SmolaVision")
    name = "extract_frames"
    description = "Extracts frames from a video at regular intervals and detects scene changes"
    inputs = {
        "video_path": {
            "type": "string",
            "description": "Path to the video file"
        },
        "interval_seconds": {
            "type": "number",
            "description": "Extract a frame every N seconds",
            "nullable": True
        },
        "detect_scenes": {
            "type": "boolean",
            "description": "Whether to detect scene changes",
            "nullable": True
        },
        "scene_threshold": {
            "type": "number",
            "description": "Threshold for scene change detection (higher = less sensitive)",
            "nullable": True
        },
        "resize_width": {
            "type": "number",
            "description": "Width to resize frames to (keeps aspect ratio)",
            "nullable": True
        },
        "start_time": {
            "type": "number",
            "description": "Start time in seconds (0 for beginning)",
            "nullable": True
        },
        "end_time": {
            "type": "number",
            "description": "End time in seconds (0 for entire video)",
            "nullable": True
        }
    }
    output_type = "array"

    def forward(self,
                video_path: str,
                interval_seconds: int = 10,
                detect_scenes: bool = True,
                scene_threshold: float = 30.0,
                resize_width: Optional[int] = None,
                start_time: float = 0.0,
                end_time: float = 0.0) -> List[Dict[str, Any]]:
        """Extract frames from a video file at regular intervals and detect scene changes"""
        logger.info(f"Extracting frames from {video_path}")
        extracted_frames = []

        try:
            # Open video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                error_msg = f"Could not open video: {video_path}"
                logger.error(error_msg)
                return [{"error": error_msg}]

            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps

            # Calculate start and end frames
            start_frame = 0
            if start_time > 0:
                start_frame = int(start_time * fps)

            end_frame = total_frames
            if end_time > 0:
                end_frame = min(int(end_time * fps), total_frames)

            actual_duration = (end_frame - start_frame) / fps

            logger.info(f"Video stats: {fps:.2f} FPS, {duration_sec:.2f} seconds total, {total_frames} frames total")
            logger.info(
                f"Processing from {start_time:.2f}s to {end_time if end_time > 0 else duration_sec:.2f}s ({actual_duration:.2f}s duration)")

            # Detect scene changes if enabled
            scene_changes = []
            if detect_scenes:
                logger.info("Detecting scene changes...")
                prev_frame = None
                last_scene_change_time = -float('inf')  # Track the last scene change time
                min_scene_duration = 1.0  # Default minimum scene duration in seconds

                # Check more frequently for scene changes (4 times per second)
                scene_check_interval = max(1, int(fps / 4))

                for frame_idx in range(start_frame, end_frame, scene_check_interval):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video.read()

                    if not success:
                        continue

                    current_time = frame_idx / fps
                        
                    # Skip if we're too close to the previous scene change
                    if current_time - last_scene_change_time < min_scene_duration:
                        prev_frame = frame.copy()  # Still update the previous frame
                        continue

                    if prev_frame is not None:
                        # Convert to grayscale for more reliable detection
                        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Calculate histograms
                        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

                        # Normalize histograms
                        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

                        # Calculate difference score using Bhattacharyya distance
                        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) * 100
                            
                        # Also calculate structural similarity for better detection
                        try:
                            from skimage.metrics import structural_similarity as ssim
                            ssim_score = ssim(gray1, gray2) * 100
                            combined_diff = (diff + (100 - ssim_score)) / 2  # Average both metrics
                        except ImportError:
                            combined_diff = diff  # Fall back to just histogram if skimage not available
                                
                        # If difference exceeds threshold, mark as scene change
                        if combined_diff > scene_threshold:
                            scene_changes.append(frame_idx)
                            last_scene_change_time = current_time
                            logger.debug(f"Scene change detected at frame {frame_idx} (time: {current_time:.2f}s, diff: {combined_diff:.2f})")

                    prev_frame = frame.copy()  # Make a copy to avoid reference issues

                    # Free memory periodically
                    if frame_idx % 1000 == 0:
                        gc.collect()

                logger.info(f"Detected {len(scene_changes)} scene changes")

            # Calculate regular interval frames (within our time range)
            frame_step = int(interval_seconds * fps)
            frames_to_extract = set(range(start_frame, end_frame, frame_step))

            # Combine with scene change frames (filtering to only those within our range)
            scene_changes = [f for f in scene_changes if start_frame <= f < end_frame]
            frames_to_extract = sorted(list(frames_to_extract.union(set(scene_changes))))
            
            # Deduplicate frames that are too similar (within regular intervals)
            if detect_scenes and len(frames_to_extract) > 1:
                logger.info("Removing duplicate frames...")
                
                # Use a lower threshold for duplicate detection than scene detection
                duplicate_threshold = scene_threshold * 0.7
                
                # Keep track of frames to remove
                frames_to_remove = set()
                
                # Compare each regular frame with nearby frames
                regular_frames = sorted(list(range(start_frame, end_frame, frame_step)))
                
                for i, frame_idx in enumerate(regular_frames):
                    # Skip if this frame is already marked for removal
                    if frame_idx in frames_to_remove:
                        continue
                        
                    # Find nearby frames (within 2 seconds)
                    nearby_frames = [f for f in frames_to_extract 
                                    if f != frame_idx and abs(f - frame_idx) < fps * 2]
                    
                    if not nearby_frames:
                        continue
                        
                    # Load the current frame
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success1, frame1 = video.read()
                    
                    if not success1:
                        continue
                        
                    # Convert to grayscale
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    
                    # Compare with nearby frames
                    for nearby_idx in nearby_frames:
                        video.set(cv2.CAP_PROP_POS_FRAMES, nearby_idx)
                        success2, frame2 = video.read()
                        
                        if not success2:
                            continue
                            
                        # Convert to grayscale
                        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate similarity
                        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                        
                        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                        
                        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) * 100
                        
                        # If frames are very similar, mark the regular frame for removal
                        # but keep scene change frames
                        if diff < duplicate_threshold and nearby_idx in scene_changes:
                            frames_to_remove.add(frame_idx)
                            logger.debug(f"Removing duplicate frame {frame_idx} (similar to scene change {nearby_idx})")
                            break
                
                # Remove duplicates
                if frames_to_remove:
                    original_count = len(frames_to_extract)
                    frames_to_extract = [f for f in frames_to_extract if f not in frames_to_remove]
                    logger.info(f"Removed {original_count - len(frames_to_extract)} duplicate frames")

            # Extract frames in batches to save memory
            batch_size = 50  # Process in smaller batches

            for i in range(0, len(frames_to_extract), batch_size):
                batch_frames = frames_to_extract[i:i + batch_size]

                for frame_idx in batch_frames:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video.read()

                    if not success:
                        logger.warning(f"Failed to read frame {frame_idx}")
                        continue

                    # Calculate timestamp
                    timestamp_sec = frame_idx / fps
                    timestamp = str(timedelta(seconds=int(timestamp_sec)))
                    safe_timestamp = timestamp.replace(":", "-")  # For safe filenames

                    # Resize if needed
                    if resize_width:
                        height, width = frame.shape[:2]
                        aspect_ratio = height / width
                        new_height = int(resize_width * aspect_ratio)
                        frame = cv2.resize(frame, (resize_width, new_height))

                    # Convert from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)

                    # Save to buffer and get base64
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=95)
                    img_bytes = buffer.getvalue()
                    img_size = len(img_bytes) / (1024 * 1024)  # Size in MB
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    # Check if this is a scene change
                    is_scene_change = frame_idx in scene_changes

                    # Add to extracted frames
                    extracted_frames.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "timestamp": timestamp,
                        "safe_timestamp": safe_timestamp,
                        "base64_image": img_base64,
                        "size_mb": img_size,
                        "is_scene_change": is_scene_change
                    })

                # Free up memory
                gc.collect()

            video.release()
            logger.info(f"Extracted {len(extracted_frames)} frames (including {len(scene_changes)} scene changes)")
            return extracted_frames

        except Exception as e:
            error_msg = f"Error extracting frames: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg}]


# Command-line interface
def main():
    """Command-line interface for SmolaVision"""
    parser = argparse.ArgumentParser(description="SmolaVision: Analyze videos using AI")

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--language", default="Hebrew", help="Language of text in the video")
    parser.add_argument("--frame-interval", type=int, default=10, help="Extract a frame every N seconds")
    parser.add_argument("--detect-scenes", action="store_true", help="Detect scene changes")
    parser.add_argument("--scene-threshold", type=float, default=30.0, help="Threshold for scene detection")
    parser.add_argument("--vision-model", default="claude", choices=["claude", "gpt4o"], help="Vision model to use")
    parser.add_argument("--summary-model", default="claude-3-5-sonnet-20240620", help="LLM for final summarization")
    parser.add_argument("--model-type", default="anthropic", choices=["anthropic", "openai", "huggingface", "ollama"],
                        help="Type of model for the agent")
    parser.add_argument("--api-key", help="API key for the model")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR text extraction from frames")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds (default: 0 = beginning)")
    parser.add_argument("--end-time", type=float, default=0.0, help="End time in seconds (default: 0 = entire video)")
    parser.add_argument("--mission", default="general", choices=["general", "workflow"],
                        help="Analysis mission type")
    parser.add_argument("--generate-flowchart", action="store_true", help="Generate a workflow flowchart")
    
    # Ollama specific arguments
    parser.add_argument("--ollama-enabled", action="store_true", help="Use Ollama for local model inference")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama API base URL")
    parser.add_argument("--ollama-model", default="llama3", help="Ollama model for text generation")
    parser.add_argument("--ollama-vision-model", default="llava", help="Ollama model for vision tasks")
    
    # Batch configuration
    parser.add_argument("--max-batch-size-mb", type=float, default=10.0, help="Maximum batch size in MB")
    parser.add_argument("--max-images-per-batch", type=int, default=15, help="Maximum images per batch")
    parser.add_argument("--batch-overlap-frames", type=int, default=2, help="Number of frames to overlap between batches")

    args = parser.parse_args()

    # Check if API key is provided or in environment
    api_key = args.api_key
    if not api_key and args.model_type != "ollama":
        if args.model_type == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif args.model_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("HF_TOKEN")

    if not api_key and args.model_type != "ollama":
        print(f"Error: No API key provided for {args.model_type} and none found in environment variables")
        return

    # Verify video file exists
    try:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
    except Exception as e:
        print(f"Error checking video file: {str(e)}")
        print("Please ensure the video path is valid and contains only supported characters.")
        return
    
    # Create configuration
    config = create_default_config(api_key)
    
    # Update model configuration
    model_config = config["model"]
    model_config.model_type = args.model_type
    model_config.vision_model = args.vision_model
    model_config.summary_model = args.summary_model
    
    # Update Ollama configuration
    model_config.ollama.enabled = args.ollama_enabled or args.model_type == "ollama"
    model_config.ollama.base_url = args.ollama_base_url
    model_config.ollama.model_name = args.ollama_model
    model_config.ollama.vision_model = args.ollama_vision_model
    
    # Update video configuration
    video_config = config["video"]
    video_config.language = args.language
    video_config.frame_interval = args.frame_interval
    video_config.detect_scenes = args.detect_scenes
    video_config.scene_threshold = args.scene_threshold
    video_config.enable_ocr = args.enable_ocr
    video_config.start_time = args.start_time
    video_config.end_time = args.end_time
    video_config.mission = args.mission
    video_config.generate_flowchart = args.generate_flowchart
    video_config.max_batch_size_mb = args.max_batch_size_mb
    video_config.max_images_per_batch = args.max_images_per_batch
    video_config.batch_overlap_frames = args.batch_overlap_frames

    # Run SmolaVision
    result = run_smolavision(video_path=args.video, config=config)

    # Print result information
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSummary of video:")
            print("-" * 80)
            print(
                result["summary_text"][:1000] + "..." if len(result["summary_text"]) > 1000 else result["summary_text"])
            print("-" * 80)
            print(f"Full summary saved to: {result['coherent_summary']}")
            print(f"Full analysis saved to: {result['full_analysis']}")

            if "flowchart" in result:
                print(f"Workflow flowchart saved to: {result['flowchart']}")
    else:
        print(result)


if __name__ == "__main__":
    main()


# Main SmolaVision Agent
def create_smolavision_agent(config: Dict[str, Any]):
    """Create the SmolaVision agent with all tools"""

    # Extract configuration
    model_config = config["model"]
    api_key = model_config.api_key
    model_type = model_config.model_type

    # Create the tools
    frame_extraction_tool = FrameExtractionTool()
    ocr_extraction_tool = OCRExtractionTool()
    batch_creation_tool = BatchCreationTool()
    vision_analysis_tool = VisionAnalysisTool()
    summarization_tool = SummarizationTool()

    # Ensure model_type is not None
    if model_type is None:
        model_type = "anthropic"
        logger.warning("model_type was None, defaulting to 'anthropic'")

    # Choose the appropriate model interface based on model_type
    if model_type == "ollama":
        # For Ollama, we'll use a local model through the API
        # The actual Ollama calls are handled in the tools
        model = LiteLLMModel(model_id="anthropic/claude-3-opus-20240229", api_key=api_key)
        logger.info("Using Ollama for local model inference (agent will use a placeholder model)")
    elif model_type == "anthropic":
        model = LiteLLMModel(model_id="anthropic/claude-3-opus-20240229", api_key=api_key)
    elif model_type == "openai":
        model = LiteLLMModel(model_id="gpt-4o", api_key=api_key)
    else:
        model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct", token=api_key)

    # Create the agent with all tools
    agent = CodeAgent(
        tools=[
            frame_extraction_tool,
            ocr_extraction_tool,
            batch_creation_tool,
            vision_analysis_tool,
            summarization_tool
        ],
        model=model,
        additional_authorized_imports=[
            "os", "json", "base64", "gc", "time", "anthropic", "openai",
            "re", "pytesseract", "PIL", "requests"
        ],
        max_steps=50,  # Allow many steps for long videos
        verbosity_level=2  # Show detailed logs
    )

    return agent


# Main function to run the entire workflow
def run_smolavision(
        video_path: str,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        frame_interval: Optional[int] = None,
        detect_scenes: Optional[bool] = None,
        scene_threshold: Optional[float] = None,
        vision_model: Optional[str] = None,
        summary_model: Optional[str] = None,
        model_type: Optional[str] = None,
        enable_ocr: Optional[bool] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        mission: Optional[str] = None,
        generate_flowchart: Optional[bool] = None,
        ollama_enabled: Optional[bool] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_vision_model: Optional[str] = None
):
    """Run the complete SmolaVision workflow"""

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create default configuration if none provided
    if config is None:
        config = create_default_config(api_key)
    
    # Override config with any explicitly provided parameters
    model_config = config.get("model", {})
    video_config = config.get("video", {})
    
    # Update model configuration
    if api_key is not None:
        model_config.api_key = api_key
    if model_type is not None:
        model_config.model_type = model_type
    if vision_model is not None:
        model_config.vision_model = vision_model
    if summary_model is not None:
        model_config.summary_model = summary_model
    
    # Update Ollama configuration
    if ollama_enabled is not None:
        model_config.ollama.enabled = ollama_enabled
    if ollama_base_url is not None:
        model_config.ollama.base_url = ollama_base_url
    if ollama_model is not None:
        model_config.ollama.model_name = ollama_model
    if ollama_vision_model is not None:
        model_config.ollama.vision_model = ollama_vision_model
    
    # Update video configuration
    if language is not None:
        video_config.language = language
    if frame_interval is not None:
        video_config.frame_interval = frame_interval
    if detect_scenes is not None:
        video_config.detect_scenes = detect_scenes
    if scene_threshold is not None:
        video_config.scene_threshold = scene_threshold
    if enable_ocr is not None:
        video_config.enable_ocr = enable_ocr
    if start_time is not None:
        video_config.start_time = start_time
    if end_time is not None:
        video_config.end_time = end_time
    if mission is not None:
        video_config.mission = mission
    if generate_flowchart is not None:
        video_config.generate_flowchart = generate_flowchart
    
    # Extract configuration values for use in the task
    api_key = model_config.api_key
    model_type = model_config.model_type
    vision_model = model_config.vision_model
    summary_model = model_config.summary_model
    ollama_config = model_config.ollama
    
    language = video_config.language
    frame_interval = video_config.frame_interval
    detect_scenes = video_config.detect_scenes
    scene_threshold = video_config.scene_threshold
    enable_ocr = video_config.enable_ocr
    start_time = video_config.start_time
    end_time = video_config.end_time
    mission = video_config.mission
    generate_flowchart = video_config.generate_flowchart
    max_batch_size_mb = video_config.max_batch_size_mb
    max_images_per_batch = video_config.max_images_per_batch
    batch_overlap_frames = video_config.batch_overlap_frames

    # Create the agent
    agent = create_smolavision_agent(config)

    # Determine which model to use for vision analysis
    vision_model_name = vision_model
    if ollama_config.enabled:
        vision_model_name = "ollama"
        logger.info(f"Using Ollama for vision analysis with model: {ollama_config.vision_model}")

    # Determine which model to use for summarization
    summary_model_name = summary_model
    if ollama_config.enabled:
        summary_model_name = "ollama"
        logger.info(f"Using Ollama for summarization with model: {ollama_config.model_name}")

    # Run the agent to process the video
    task = f"""
Analyze the video at path "{video_path}" with the following steps:

1. Extract frames from the video using extract_frames tool
   - Use interval of {frame_interval} seconds
   - {'Detect scene changes' if detect_scenes else 'Do not detect scene changes'}
   - Use scene threshold of {scene_threshold}
   - Start at {start_time} seconds
   - {'End at ' + str(end_time) + ' seconds' if end_time > 0 else 'Process until the end of the video'}

2. {'Apply OCR to extract text from frames using extract_text_ocr tool' if enable_ocr else '# OCR disabled'}
   {'- Specify language as "' + language + '"' if enable_ocr else ''}

3. Create batches of frames using create_batches tool
   - Use maximum batch size of {max_batch_size_mb} MB
   - Use maximum of {max_images_per_batch} images per batch
   - Use {batch_overlap_frames} frames of overlap between batches

4. Analyze each batch using analyze_batch tool
   - Pass context from previous batch to maintain continuity
   - Specify language as "{language}"
   - Use "{vision_model_name}" as the vision model
   {'- Use mission type "' + mission + '"' if mission != "general" else ''}
   {'- Make use of OCR text in frames when available' if enable_ocr else ''}

5. Generate a final coherent summary using generate_summary tool
   - Combine all the analyses
   - Use "{summary_model_name}" as the summary model
   {'- Use mission type "' + mission + '"' if mission != "general" else ''}
   {'- Generate a workflow flowchart' if generate_flowchart else ''}

The goal is to create a comprehensive analysis of the video, capturing all visual elements,
text content (with translations), and {'workflow logic' if mission == 'workflow' else 'narrative flow'}.
"""

    logger.info(f"Starting analysis of video: {video_path}")
    start_time_execution = time.time()

    try:
        # Run the agent
        result = agent.run(task, additional_args={
            "api_key": api_key,
            "language": language,
            "vision_model": vision_model_name,
            "summary_model": summary_model_name,
            "enable_ocr": enable_ocr,
            "start_time": start_time,
            "end_time": end_time,
            "mission": mission,
            "generate_flowchart": generate_flowchart,
            "ollama_config": ollama_config.to_dict() if ollama_config.enabled else None
        })

        # Log completion
        elapsed_time = time.time() - start_time_execution
        logger.info(f"Analysis completed in {elapsed_time:.1f} seconds")
        logger.info(f"Results saved to directory: {output_dir}")

        return result

    except Exception as e:
        logger.error(f"Error running SmolaVision: {str(e)}")
        return f"Error: {str(e)}"


# Command-line interface
def main():
    """Command-line interface for SmolaVision"""
    parser = argparse.ArgumentParser(description="SmolaVision: Analyze videos using AI")

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--language", default="Hebrew", help="Language of text in the video")
    parser.add_argument("--frame-interval", type=int, default=10, help="Extract a frame every N seconds")
    parser.add_argument("--detect-scenes", action="store_true", help="Detect scene changes")
    parser.add_argument("--scene-threshold", type=float, default=30.0, help="Threshold for scene detection")
    parser.add_argument("--vision-model", default="claude", choices=["claude", "gpt4o"], help="Vision model to use")
    parser.add_argument("--summary-model", default="claude-3-5-sonnet-20240620", help="LLM for final summarization")
    parser.add_argument("--model-type", default="anthropic", choices=["anthropic", "openai", "huggingface", "ollama"],
                        help="Type of model for the agent")
    parser.add_argument("--api-key", help="API key for the model")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR text extraction from frames")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds (default: 0 = beginning)")
    parser.add_argument("--end-time", type=float, default=0.0, help="End time in seconds (default: 0 = entire video)")
    parser.add_argument("--mission", default="general", choices=["general", "workflow"],
                        help="Analysis mission type")
    parser.add_argument("--generate-flowchart", action="store_true", help="Generate a workflow flowchart")
    
    # Ollama specific arguments
    parser.add_argument("--ollama-enabled", action="store_true", help="Use Ollama for local model inference")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama API base URL")
    parser.add_argument("--ollama-model", default="llama3", help="Ollama model for text generation")
    parser.add_argument("--ollama-vision-model", default="llava", help="Ollama model for vision tasks")
    
    # Batch configuration
    parser.add_argument("--max-batch-size-mb", type=float, default=10.0, help="Maximum batch size in MB")
    parser.add_argument("--max-images-per-batch", type=int, default=15, help="Maximum images per batch")
    parser.add_argument("--batch-overlap-frames", type=int, default=2, help="Number of frames to overlap between batches")

    args = parser.parse_args()

    # Check if API key is provided or in environment
    api_key = args.api_key
    if not api_key and args.model_type != "ollama":
        if args.model_type == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif args.model_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("HF_TOKEN")

    if not api_key and args.model_type != "ollama":
        print(f"Error: No API key provided for {args.model_type} and none found in environment variables")
        return

    # Verify video file exists
    try:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
    except Exception as e:
        print(f"Error checking video file: {str(e)}")
        print("Please ensure the video path is valid and contains only supported characters.")
        return
    
    # Create configuration
    config = create_default_config(api_key)
    
    # Update model configuration
    model_config = config["model"]
    model_config.model_type = args.model_type
    model_config.vision_model = args.vision_model
    model_config.summary_model = args.summary_model
    
    # Update Ollama configuration
    model_config.ollama.enabled = args.ollama_enabled or args.model_type == "ollama"
    model_config.ollama.base_url = args.ollama_base_url
    model_config.ollama.model_name = args.ollama_model
    model_config.ollama.vision_model = args.ollama_vision_model
    
    # Update video configuration
    video_config = config["video"]
    video_config.language = args.language
    video_config.frame_interval = args.frame_interval
    video_config.detect_scenes = args.detect_scenes
    video_config.scene_threshold = args.scene_threshold
    video_config.enable_ocr = args.enable_ocr
    video_config.start_time = args.start_time
    video_config.end_time = args.end_time
    video_config.mission = args.mission
    video_config.generate_flowchart = args.generate_flowchart
    video_config.max_batch_size_mb = args.max_batch_size_mb
    video_config.max_images_per_batch = args.max_images_per_batch
    video_config.batch_overlap_frames = args.batch_overlap_frames

    # Run SmolaVision
    result = run_smolavision(video_path=args.video, config=config)

    # Print result information
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSummary of video:")
            print("-" * 80)
            print(
                result["summary_text"][:1000] + "..." if len(result["summary_text"]) > 1000 else result["summary_text"])
            print("-" * 80)
            print(f"Full summary saved to: {result['coherent_summary']}")
            print(f"Full analysis saved to: {result['full_analysis']}")

            if "flowchart" in result:
                print(f"Workflow flowchart saved to: {result['flowchart']}")
    else:
        print(result)


if __name__ == "__main__":
    main()
