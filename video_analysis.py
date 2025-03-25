#!/usr/bin/env python
"""
SmolaVision: A system for analyzing videos using smolagents and vision models.
Process a video by extracting frames, sending to vision AI models, and creating a cohesive summary.
"""

import os
import cv2
import base64
import time
import json
import numpy as np
import argparse
import logging
import asyncio
import gc
import base64
import dotenv
from io import BytesIO
from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Import computer vision libraries
import cv2
from PIL import Image
from io import BytesIO

# Import smolagents
from smolagents import CodeAgent, Tool, HfApiModel, LiteLLMModel
import ollama

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


# ===== TOOL DEFINITIONS =====

class FrameExtractionTool(Tool):
    """Tool for extracting frames from a video file"""
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

                # Check twice per second for scene changes
                scene_check_interval = max(1, int(fps / 2))

                for frame_idx in range(start_frame, end_frame, scene_check_interval):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video.read()

                    if not success:
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

                        # Calculate difference score
                        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) * 100

                        # If difference exceeds threshold, mark as scene change
                        if diff > scene_threshold:
                            scene_changes.append(frame_idx)
                            logger.debug(f"Scene change detected at frame {frame_idx} (diff: {diff:.2f})")

                    prev_frame = frame

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


class OCRExtractionTool(Tool):
    """Tool for extracting text from frames using OCR"""
    name = "extract_text_ocr"
    description = "Extracts text from frames using OCR with support for Hebrew"
    inputs = {
        "frames": {
            "type": "array",
            "description": "List of extracted frames to process with OCR"
        },
        "language": {
            "type": "string",
            "description": "Primary language to optimize OCR for (e.g., 'heb' for Hebrew)",
            "nullable": True
        }
    }
    output_type = "array"

    def forward(self, frames: List[Dict], language: str = "heb") -> List[Dict]:
        """Extract text from frames using Tesseract OCR"""
        logger.info(f"Extracting text using OCR from {len(frames)} frames")

        try:
            import pytesseract
            from PIL import Image
            import base64
            from io import BytesIO
            import re

            # Map common language names to Tesseract language codes
            language_map = {
                "Hebrew": "heb",
                "English": "eng",
                "Arabic": "ara",
                "Russian": "rus",
                # Add more mappings as needed
            }

            # Ensure language is not None before using it
            if language is None:
                language = "eng"
                logger.warning("Language parameter was None, defaulting to English")

            # Get the Tesseract language code
            lang_code = language_map.get(language, language)

            # Process each frame with OCR
            for i, frame in enumerate(frames):
                if "base64_image" not in frame:
                    logger.warning(f"Frame {i} missing base64_image, skipping OCR")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = "Missing base64_image"
                    continue

                # Ensure the base64_image is a string before decoding
                base64_image = frame.get("base64_image", "")
                if not isinstance(base64_image, str) or not base64_image:
                    logger.warning(f"Frame {i} has invalid base64_image data, skipping OCR")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = "Invalid base64 image data"
                    continue

                try:
                    # Decode base64 image
                    img_data = base64.b64decode(base64_image)
                    img = Image.open(BytesIO(img_data))

                    # Run OCR with specified language
                    text = pytesseract.image_to_string(img, lang=lang_code)

                    # Clean up text (remove excessive whitespace)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Add the extracted text to the frame data
                    frame["ocr_text"] = text
                    frame["ocr_language"] = lang_code

                    # Log a preview of the text (first 50 chars)
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    logger.debug(f"Frame {i} OCR: {text_preview}")

                except Exception as e:
                    logger.error(f"OCR failed for frame {i}: {str(e)}")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = str(e)

            logger.info(f"OCR extraction completed for {len(frames)} frames")
            return frames

        except ImportError:
            logger.error("Required OCR libraries not installed. Install with: pip install pytesseract pillow")
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_error"] = "OCR libraries not installed"
            return frames
        except Exception as e:
            error_msg = f"Error during OCR processing: {str(e)}"
            logger.error(error_msg)
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_error"] = error_msg
            return frames


class BatchCreationTool(Tool):
    """Tool for creating batches of frames for processing"""
    name = "create_batches"
    description = "Groups frames into batches based on size and count limits"
    inputs = {
        "frames": {
            "type": "array",
            "description": "List of extracted frames"
        },
        "max_batch_size_mb": {
            "type": "number",
            "description": "Maximum size of a batch in MB",
            "nullable": True
        },
        "max_images_per_batch": {
            "type": "number",
            "description": "Maximum number of images in a batch",
            "nullable": True
        },
        "overlap_frames": {
            "type": "number",
            "description": "Number of frames to overlap between batches",
            "nullable": True
        }
    }
    output_type = "array"

    def forward(self, frames: List[Dict],
                max_batch_size_mb: float = 10.0,
                max_images_per_batch: int = 15,
                overlap_frames: int = 2) -> List[List[Dict]]:
        """Group frames into batches based on size and count limits"""
        logger.info("Creating batches from extracted frames")

        try:
            batches = []
            current_batch = []
            current_batch_size = 0

            # Sort frames by timestamp to ensure chronological order
            sorted_frames = sorted(frames, key=lambda x: x.get("timestamp_sec", 0))

            for i, frame in enumerate(sorted_frames):
                # Always start a new batch at scene changes if not the first frame
                if frame.get("is_scene_change", False) and current_batch and i > 0:
                    # If this frame would make the batch too large, finish the current batch first
                    if (current_batch_size + frame.get("size_mb", 0) > max_batch_size_mb or
                            len(current_batch) >= max_images_per_batch):
                        batches.append(current_batch.copy())

                        # Start new batch with overlap, including the scene change frame
                        overlap_start = max(0, len(current_batch) - overlap_frames)
                        current_batch = current_batch[overlap_start:]
                        current_batch_size = sum(f.get("size_mb", 0) for f in current_batch)

                # If adding this frame would exceed limits, finalize the current batch
                if (current_batch_size + frame.get("size_mb", 0) > max_batch_size_mb or
                    len(current_batch) >= max_images_per_batch) and current_batch:
                    batches.append(current_batch.copy())

                    # Start new batch with overlap
                    overlap_start = max(0, len(current_batch) - overlap_frames)
                    current_batch = current_batch[overlap_start:]
                    current_batch_size = sum(f.get("size_mb", 0) for f in current_batch)

                # Add frame to current batch
                current_batch.append(frame)
                current_batch_size += frame.get("size_mb", 0)

            # Add the last batch if it's not empty
            if current_batch:
                batches.append(current_batch)

            logger.info(f"Created {len(batches)} batches from {len(frames)} frames")

            # Log batch statistics
            batch_sizes = [len(batch) for batch in batches]
            if batch_sizes:
                logger.info(f"Batch statistics: min={min(batch_sizes)}, max={max(batch_sizes)}, "
                            f"avg={sum(batch_sizes) / len(batch_sizes):.1f} frames per batch")

            return batches

        except Exception as e:
            error_msg = f"Error creating batches: {str(e)}"
            logger.error(error_msg)
            return [[{"error": error_msg}]]


class VisionAnalysisTool(Tool):
    """Tool for analyzing batches of frames using a vision model"""
    name = "analyze_batch"
    description = "Analyzes a batch of frames using a vision model"
    inputs = {
        "batch": {
            "type": "array",
            "description": "Batch of frames to analyze"
        },
        "previous_context": {
            "type": "string",
            "description": "Context from previous analysis for continuity",
            "nullable": True
        },
        "language": {
            "type": "string",
            "description": "Language of text in the video",
            "nullable": True
        },
        "model_name": {
            "type": "string",
            "description": "Name of the vision model to use (claude or gpt4o)",
            "nullable": True
        },
        "api_key": {
            "type": "string",
            "description": "API key for the vision model",
            "nullable": True
        },
        "mission": {
            "type": "string",
            "description": "Specific analysis mission (e.g., 'workflow', 'general')",
            "nullable": True
        },
        "ollama_config": {
            "type": "object",
            "description": "Configuration for Ollama local models",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, batch: List[Dict],
                previous_context: str = "",
                language: str = "Hebrew",
                model_name: str = "claude",
                api_key: str = "",
                mission: str = "general",
                ollama_config: Optional[Dict] = None) -> str:
        """Analyze a batch of frames using a vision model"""
        if not batch or len(batch) == 0:
            return "Error: Empty batch received"

        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Ensure model_name is not None
        if model_name is None:
            model_name = "claude"
            logger.warning("model_name was None, defaulting to 'claude'")

        # Get first and last timestamp for logging
        start_time = batch[0].get("timestamp", "unknown")
        end_time = batch[-1].get("timestamp", "unknown")

        logger.info(f"Analyzing batch from {start_time} to {end_time} ({len(batch)} frames)")

        try:
            # Create prompt based on mission type
            if mission and mission.lower() == "workflow":
                prompt = self._create_workflow_prompt(batch, previous_context, language)
            else:
                prompt = self._create_general_prompt(batch, previous_context, language)

            # Check if using Ollama
            if model_name == "ollama" or (ollama_config and ollama_config.get("enabled")):
                try:
                    # Use Ollama for local inference
                    base_url = ollama_config.get("base_url", "http://localhost:11434") if ollama_config else "http://localhost:11434"
                    vision_model = ollama_config.get("vision_model", "llava") if ollama_config else "llava"

                    # Create or reuse Ollama client
                    try:
                        if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                            self._ollama_client = OllamaClient(base_url=base_url)

                        # For smaller GPUs, we need to be careful with batch size
                        # Process images in smaller sub-batches if needed
                        max_images_per_request = 3  # Limit for 12GB VRAM
                    except Exception as e:
                        logger.error(f"Error initializing Ollama client: {str(e)}")
                        return f"Error initializing Ollama client: {str(e)}"

                    if len(batch) > max_images_per_request:
                        logger.info(f"Batch size ({len(batch)}) exceeds max images per request ({max_images_per_request}). Processing in sub-batches.")

                        # Process in sub-batches and combine results
                        sub_batch_results = []
                        for i in range(0, len(batch), max_images_per_request):
                            sub_batch = batch[i:i + max_images_per_request]

                            # Extract base64 images for this sub-batch
                            images = []
                            for frame in sub_batch:
                                base64_image = frame.get('base64_image', '')
                                if base64_image and isinstance(base64_image, str):
                                    images.append(base64_image)

                            # Create a sub-prompt
                            sub_prompt = f"Analyzing frames {i+1} to {i+len(sub_batch)} of {len(batch)}:\n{prompt}"

                            try:
                                # Call Ollama vision model for this sub-batch
                                sub_result = self._ollama_client.generate_vision(
                                    model=vision_model,
                                    prompt=sub_prompt,
                                    images=images,
                                    max_tokens=2048  # Smaller token limit for sub-batches
                                )

                                sub_batch_results.append(sub_result)
                            except Exception as e:
                                error_msg = f"Error processing sub-batch {i+1}: {str(e)}"
                                logger.error(error_msg)
                                sub_batch_results.append(error_msg)

                        # Combine results
                        analysis = "\n\n".join(sub_batch_results)
                    else:
                        # Process the whole batch at once
                        # Extract base64 images for Ollama
                        images = []
                        for frame in batch:
                            base64_image = frame.get('base64_image', '')
                            if base64_image and isinstance(base64_image, str):
                                images.append(base64_image)

                        try:
                            # Call Ollama vision model
                            response = self._ollama_client.generate_vision(
                                model=vision_model,
                                prompt=prompt,
                                images=images,
                                max_tokens=4096
                            )

                            # IMPORTANT: Always ensure we return a string, not an object
                            # Extract content safely from any response format
                            try:
                                # Log the response type for debugging
                                logger.debug(f"Vision response type: {type(response)}")

                                # Try object attribute access first (newer Ollama versions)
                                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                                    analysis = str(response.message.content)
                                # Try dictionary access (older Ollama versions)
                                elif isinstance(response, dict):
                                    if 'message' in response:
                                        if isinstance(response['message'], dict) and 'content' in response['message']:
                                            analysis = str(response['message']['content'])
                                        elif hasattr(response['message'], 'content'):
                                            analysis = str(response['message'].content)
                                        else:
                                            analysis = str(response['message'])
                                # Try direct content attribute
                                elif hasattr(response, 'content'):
                                    analysis = str(response.content)
                                # Try dictionary content key
                                elif isinstance(response, dict) and 'content' in response:
                                    analysis = str(response['content'])
                                # If it's already a string, use it directly
                                elif isinstance(response, str):
                                    analysis = response
                                # Last resort: convert to string
                                else:
                                    analysis = str(response)
                            except Exception as e:
                                logger.error(f"Error extracting content from vision response: {str(e)}")
                                analysis = f"Error extracting content from vision response: {str(e)}"
                        except Exception as e:
                            error_msg = f"Error calling Ollama vision model: {str(e)}"
                            logger.error(error_msg)
                            analysis = error_msg
                except Exception as e:
                    error_msg = f"Error in Ollama vision processing: {str(e)}"
                    logger.error(error_msg)
                    analysis = error_msg

            # Prepare images based on the model
            elif model_name.startswith("claude"):
                try:
                    # Anthropic Claude format
                    import anthropic

                    # Create client (or reuse cached one)
                    if not hasattr(self, '_anthropic_client') or self._anthropic_client is None:
                        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

                    # Format images for Claude
                    images = []
                    for frame in batch:
                        base64_image = frame.get('base64_image', '')
                        if base64_image and isinstance(base64_image, str):
                            images.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            })

                    # Make API call
                    model_id = "claude-3-opus-20240229" if model_name == "claude" else model_name

                    # Create the content array for the message
                    message_content = [
                        {"type": "text", "text": prompt}
                    ]
                    message_content.extend(images)

                    # Make the API call with properly formatted content
                    response = self._anthropic_client.messages.create(
                        model=model_id,
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": message_content
                            }
                        ]
                    )
                    analysis = response.content[0].text
                except (ImportError, ModuleNotFoundError):
                    logger.error("Anthropic API not available. Please install with: pip install anthropic")
                    return "Error: Anthropic API not available. Please install with: pip install anthropic"
                except Exception as e:
                    logger.error(f"Error using Anthropic API: {str(e)}")
                    return f"Error using Anthropic API: {str(e)}"

            elif model_name == "gpt4o" or model_name == "gpt-4o":
                try:
                    # OpenAI GPT-4 Vision format
                    import openai

                    # Create client (or reuse cached one)
                    if not hasattr(self, '_openai_client') or self._openai_client is None:
                        self._openai_client = openai.OpenAI(api_key=api_key)

                    # Format images for OpenAI
                    images = []
                    for frame in batch:
                        base64_image = frame.get('base64_image', '')
                        if base64_image and isinstance(base64_image, str):
                            images.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })

                    # Make API call
                    response = self._openai_client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    *images
                                ]
                            }
                        ]
                    )
                    analysis = response.choices[0].message.content
                except (ImportError, ModuleNotFoundError):
                    logger.error("OpenAI API not available. Please install with: pip install openai")
                    return "Error: OpenAI API not available. Please install with: pip install openai"
                except Exception as e:
                    logger.error(f"Error using OpenAI API: {str(e)}")
                    return f"Error using OpenAI API: {str(e)}"

            else:
                return f"Error: Unsupported model name: {model_name}. Use 'claude', 'gpt4o', or 'ollama'."

            logger.info(f"Successfully analyzed batch from {start_time} to {end_time}")

            # Save analysis to a file
            output_dir = os.path.join("output", "batch_analyses")
            os.makedirs(output_dir, exist_ok=True)

            safe_start = batch[0].get("safe_timestamp", "unknown")
            safe_end = batch[-1].get("safe_timestamp", "unknown")

            analysis_file = os.path.join(output_dir, f"batch_{safe_start}_to_{safe_end}.txt")
            with open(analysis_file, "w", encoding="utf-8") as f:
                # Add batch metadata as a header
                f.write(f"# Batch Analysis: {start_time} to {end_time}\n")
                f.write(f"# Frames: {len(batch)}\n")
                f.write(f"# Mission: {mission}\n")
                f.write(
                    f"# Scene changes: {len([frame for frame in batch if frame.get('is_scene_change', False)])}\n\n")
                f.write(analysis)

            return analysis

        except Exception as e:
            error_msg = f"Error analyzing batch from {start_time} to {end_time}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _create_general_prompt(self, batch, previous_context, language):
        """Create the standard general analysis prompt"""
        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Prepare timestamps, noting scene changes
        timestamp_info = []
        for frame in batch:
            if frame.get("is_scene_change", False):
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}*]")  # Mark scene changes with asterisk
            else:
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}]")

        timestamps_str = ", ".join(timestamp_info)

        # Add indicator for scene changes if any exist in this batch
        scene_changes = [frame for frame in batch if frame.get("is_scene_change", False)]
        scene_change_note = ""
        if scene_changes:
            scene_change_timestamps = [f"[{frame.get('timestamp', 'unknown')}]" for frame in scene_changes]
            scene_change_note = (f"\nNOTE: Frames marked with * represent detected scene changes at "
                                 f"{', '.join(scene_change_timestamps)}. Pay special attention to changes "
                                 f"in content, setting, or context at these points.")

        # Include OCR text if available
        ocr_texts = []
        for frame in batch:
            if frame.get("ocr_text"):
                ocr_texts.append(f"[{frame.get('timestamp', 'unknown')}]: {frame.get('ocr_text')}")

        ocr_section = ""
        if ocr_texts:
            ocr_section = "\n\nExtracted text from OCR:\n" + "\n".join(ocr_texts)

        # Prepare context
        if previous_context:
            context_intro = f"\n\nContext from previous video segments:\n{previous_context}\n\n"
        else:
            context_intro = ""

        # Create prompt
        prompt = f"""
{context_intro}
Analyze the following video frames captured at timestamps {timestamps_str}. The video is in {language} language.{scene_change_note}{ocr_section}

Key instructions:
1. Describe all visual elements, text, people, actions, and settings
2. Transcribe and translate any {language} text visible in the frames
3. Note any transitions or changes across frames, especially at scene change points
4. Maintain continuity with the previous context provided (if any)
5. If text appears to continue from previous frames or context, attempt to complete the meaning

Organize your analysis chronologically, noting timestamps where appropriate.
Be detailed and comprehensive in your analysis.
"""
        return prompt

    def _create_workflow_prompt(self, batch, previous_context, language):
        """Create a specialized prompt for workflow analysis"""
        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Prepare timestamps, noting scene changes
        timestamp_info = []
        for frame in batch:
            if frame.get("is_scene_change", False):
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}*]")  # Mark scene changes with asterisk
            else:
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}]")

        timestamps_str = ", ".join(timestamp_info)

        # Check for scene changes
        scene_changes = [frame for frame in batch if frame.get("is_scene_change", False)]
        scene_change_note = ""
        if scene_changes:
            scene_change_timestamps = [f"[{frame.get('timestamp', 'unknown')}]" for frame in scene_changes]
            scene_change_note = (f"\nNOTE: Frames marked with * represent detected scene changes at "
                                 f"{', '.join(scene_change_timestamps)}. Pay special attention to workflow transitions at these points.")

        # Include OCR text if available
        ocr_texts = []
        for frame in batch:
            if frame.get("ocr_text"):
                ocr_texts.append(f"[{frame.get('timestamp', 'unknown')}]: {frame.get('ocr_text')}")

        ocr_section = ""
        if ocr_texts:
            ocr_section = "\n\nExtracted text from OCR:\n" + "\n".join(ocr_texts)

        # Prepare context
        if previous_context:
            context_intro = f"\n\nContext from previous video segments:\n{previous_context}\n\n"
        else:
            context_intro = ""

        # Create workflow-specific prompt
        prompt = f"""
{context_intro}
Analyze the following video frames captured at timestamps {timestamps_str}. These frames show interactions with an AI platform.{scene_change_note}{ocr_section}

Key instructions for workflow analysis:
1. Identify user roles and AI roles in the interaction (who is doing what)
2. Analyze the prompting patterns and styles used with the AI
3. Identify key steps in the workflow of interacting with the AI platform
4. Note the sequence of actions, inputs, and outputs
5. Identify any UI elements and their functions in the workflow
6. Note any specific "prompt engineering" techniques visible
7. Maintain continuity with the previous context provided (if any)
8. If text appears to continue from previous frames or context, attempt to complete the meaning

Organize your analysis chronologically, focusing on the workflow logic and interaction patterns.
Your analysis will later be used to create a flow diagram of the entire process.
"""
        return prompt


class SummarizationTool(Tool):
    """Tool for generating a coherent summary from batch analyses"""
    name = "generate_summary"
    description = "Creates a coherent summary from all batch analyses"
    inputs = {
        "analyses": {
            "type": "array",
            "description": "List of all batch analyses"
        },
        "language": {
            "type": "string",
            "description": "Language of text in the video",
            "nullable": True
        },
        "model_name": {
            "type": "string",
            "description": "Name of the LLM to use for summarization",
            "nullable": True
        },
        "api_key": {
            "type": "string",
            "description": "API key for the LLM",
            "nullable": True
        },
        "mission": {
            "type": "string",
            "description": "Specific analysis mission (e.g., 'workflow', 'general')",
            "nullable": True
        },
        "generate_flowchart": {
            "type": "boolean",
            "description": "Whether to generate a flowchart diagram",
            "nullable": True
        },
        "ollama_config": {
            "type": "object",
            "description": "Configuration for Ollama local models",
            "nullable": True
        }
    }
    output_type = "object"

    def forward(self, analyses: List[str],
                language: str = "Hebrew",
                model_name: str = "claude-3-5-sonnet-20240620",
                api_key: str = "",
                mission: str = "general",
                generate_flowchart: bool = False,
                ollama_config: Optional[Dict] = None) -> Dict:
        """Generate a coherent summary from all batch analyses"""
        logger.info("Generating final summary from all analyses")

        # Check for empty or None analyses
        if not analyses:
            return {"error": "No analyses provided"}

        # Filter out None values
        analyses = [analysis for analysis in analyses if analysis is not None]
        if not analyses:
            return {"error": "All analyses were None"}

        try:
            # Ensure language is not None
            if language is None:
                language = "English"
                logger.warning("Language parameter was None, defaulting to English")

            # Ensure model_name is not None
            if model_name is None:
                model_name = "claude-3-5-sonnet-20240620"
                logger.warning("model_name was None, defaulting to default model")

            # First, concatenate all analyses for the full detailed version
            full_analysis = "\n\n---\n\n".join(analyses)

            # Save the full analysis to a file
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            full_analysis_path = os.path.join(output_dir, "video_analysis_full.txt")
            with open(full_analysis_path, "w", encoding="utf-8") as f:
                f.write(full_analysis)

            logger.info(f"Saved full analysis to {full_analysis_path}")

            # Generate a coherent summary using the specified LLM
            logger.info(f"Generating coherent narrative summary using {model_name}")

            # Split the full analysis into manageable chunks for the LLM
            max_chunk_size = 12000
            chunks = self._chunk_text(full_analysis, max_chunk_size)
            logger.info(f"Split analysis into {len(chunks)} chunks for summarization")

            # Choose prompt template based on mission type and flowchart request
            if mission and mission.lower() == "workflow" and generate_flowchart:
                prompt_template = self._workflow_with_flowchart_prompt_template
            elif mission and mission.lower() == "workflow":
                prompt_template = self._workflow_prompt_template
            else:
                prompt_template = self._general_prompt_template

            # Process each chunk to build a complete summary
            complete_summary = ""

            for i, chunk in enumerate(chunks):
                is_first = (i == 0)
                is_last = (i == len(chunks) - 1)

                logger.info(f"Generating summary for chunk {i + 1}/{len(chunks)}")

                # Create a prompt based on chunk position
                if is_first and is_last:  # Only one chunk
                    prompt = prompt_template.format(language=language, chunk=chunk)
                elif is_first:  # First of multiple chunks
                    prompt = f"""
You are analyzing a video in {language}. Below is the first part of a detailed analysis of the video frames.

Please begin creating a well-structured summary. This is part 1 of a multi-part summary process.
Focus on:
1. Describing the key visual elements, settings, people, and actions
2. Including all important {language} text with translations
3. Maintaining chronological flow
4. Setting up context for later parts of the video

The goal is to start a cohesive narrative that will be continued with additional content.

VIDEO ANALYSIS (PART 1):
{chunk}
"""
                elif is_last:  # Last of multiple chunks
                    prompt = f"""
You are continuing to analyze a video in {language}. Below is the final part of a detailed analysis of the video frames.

This is the final part in a multi-part summary process. You've already summarized earlier parts as follows:

PREVIOUS SUMMARY:
{complete_summary}

Please complete the summary by:
1. Continuing the narrative from where the previous summary left off
2. Integrating new information from this final section
3. Ensuring all important {language} text is included with translations
4. Creating proper closure and concluding the summary
5. Maintaining consistency with the style and approach of the previous summary

VIDEO ANALYSIS (FINAL PART):
{chunk}
"""
                    # If this is workflow with flowchart and we're on the last chunk, add flowchart instructions
                    if mission and mission.lower() == "workflow" and generate_flowchart:
                        prompt += """
After completing the narrative summary, please create a section titled "Workflow Diagram" containing a Mermaid flowchart that visualizes the workflow.
Use this format for the Mermaid diagram:
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
    %% Add all needed nodes and connections to represent the workflow
```
"""
                else:  # Middle chunk
                    prompt = f"""
You are continuing to analyze a video in {language}. Below is a middle part of a detailed analysis of the video frames.

This is a continuation of a multi-part summary process. You've already summarized earlier parts as follows:

PREVIOUS SUMMARY:
{complete_summary}

Please continue the summary by:
1. Picking up where the previous summary left off
2. Integrating new information from this section
3. Ensuring all important {language} text is included with translations
4. Maintaining chronological flow and narrative coherence
5. Setting up context for later parts of the video

VIDEO ANALYSIS (MIDDLE PART):
{chunk}
"""

                # Check if using Ollama
                if model_name == "ollama" or (ollama_config and ollama_config.get("enabled")):
                    try:
                        # Use Ollama for local inference
                        base_url = ollama_config.get("base_url", "http://localhost:11434") if ollama_config else "http://localhost:11434"
                        model = ollama_config.get("model_name", "llama3") if ollama_config else "llama3"

                        # Create or reuse Ollama client
                        if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                            self._ollama_client = OllamaClient(base_url=base_url)
                    except Exception as e:
                        logger.error(f"Error initializing Ollama client: {str(e)}")
                        return {"error": f"Error initializing Ollama client: {str(e)}"}

                    # For smaller models, we need to be more careful with context length
                    # Determine if we should use a smaller model for better performance
                    if len(prompt) > 8000 and "small_models" in ollama_config:
                        # Use the smallest model for very large prompts
                        logger.info(f"Using smaller model for large prompt ({len(prompt)} chars)")
                        model = ollama_config.get("small_models", {}).get("fast", model)

                    # Call Ollama model with appropriate token limit
                    # Smaller models need smaller token limits
                    max_tokens = 2048 if "phi" in model or "gemma:2b" in model or "tiny" in model else 4096

                    try:
                        chunk_summary = self._ollama_client.generate(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens
                        )
                    except Exception as e:
                        error_msg = f"Error calling Ollama API: {str(e)}"
                        logger.error(error_msg)
                        chunk_summary = error_msg

                # Call the LLM API based on model name
                elif model_name.startswith("claude"):
                    try:
                        import anthropic

                        # Create or reuse client
                        if not hasattr(self, '_anthropic_client') or self._anthropic_client is None:
                            self._anthropic_client = anthropic.Anthropic(api_key=api_key)

                        response = self._anthropic_client.messages.create(
                            model=model_name,
                            max_tokens=4096,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )
                        chunk_summary = response.content[0].text
                    except (ImportError, ModuleNotFoundError):
                        logger.error("Anthropic API not available. Please install with: pip install anthropic")
                        return {"error": "Anthropic API not available. Please install with: pip install anthropic"}
                    except Exception as e:
                        logger.error(f"Error using Anthropic API: {str(e)}")
                        return {"error": f"Error using Anthropic API: {str(e)}"}

                elif model_name.startswith("gpt"):
                    try:
                        import openai

                        # Create or reuse client
                        if not hasattr(self, '_openai_client') or self._openai_client is None:
                            self._openai_client = openai.OpenAI(api_key=api_key)

                        response = self._openai_client.chat.completions.create(
                            model=model_name,
                            max_tokens=4096,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )
                        chunk_summary = response.choices[0].message.content
                    except (ImportError, ModuleNotFoundError):
                        logger.error("OpenAI API not available. Please install with: pip install openai")
                        return {"error": "OpenAI API not available. Please install with: pip install openai"}
                    except Exception as e:
                        logger.error(f"Error using OpenAI API: {str(e)}")
                        return {"error": f"Error using OpenAI API: {str(e)}"}

                else:
                    return {"error": f"Unsupported model: {model_name}"}

                # Update the complete summary
                if is_first:
                    complete_summary = chunk_summary
                else:
                    # For non-first chunks, append the new summary
                    complete_summary += "\n\n" + chunk_summary

            # If flowchart was requested and generated, save it
            flowchart_path = None
            if generate_flowchart and mission and mission.lower() == "workflow":
                # Extract the flowchart Mermaid code
                import re
                flowchart_pattern = r"```mermaid\s*([\s\S]*?)\s*```"
                flowchart_match = re.search(flowchart_pattern, complete_summary)

                if flowchart_match:
                    flowchart_code = flowchart_match.group(1).strip()
                    flowchart_path = os.path.join(output_dir, "workflow_flowchart.mmd")

                    with open(flowchart_path, "w", encoding="utf-8") as f:
                        f.write(flowchart_code)

                    logger.info(f"Saved workflow flowchart to {flowchart_path}")

            # Save the coherent summary to a file
            summary_path = os.path.join(output_dir, "video_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(complete_summary)

            logger.info(f"Saved coherent summary to {summary_path}")

            # Return results including flowchart if generated
            result = {
                "full_analysis": full_analysis_path,
                "coherent_summary": summary_path,
                "summary_text": complete_summary
            }

            if flowchart_path:
                result["flowchart"] = flowchart_path

            return result

        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _chunk_text(self, text: str, max_chunk_size: int = 12000) -> List[str]:
        """Split text into chunks for processing by LLM"""
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, start a new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Templates for different types of summaries
    _general_prompt_template = """
You are analyzing a video in {language}. Below is a detailed analysis of the video frames.

Please create a well-structured, comprehensive summary of the entire video. This summary should:
1. Describe the key visual elements, settings, people, and actions
2. Include all important {language} text with translations
3. Maintain chronological flow and narrative coherence
4. Highlight main topics or themes
5. Be detailed while eliminating redundancy

The goal is to create a cohesive narrative that someone could read to understand the full content of the video.

VIDEO ANALYSIS:
{chunk}
"""

    _workflow_prompt_template = """
You are analyzing a video that demonstrates workflow interactions with an AI platform. Below is a detailed analysis of the video frames.

Please create a well-structured, comprehensive summary of the AI interaction workflow shown in the video. This summary should:
1. Identify distinct roles in the interaction (user types, AI roles)
2. Describe the key steps in the workflow sequence
3. Analyze prompting patterns and strategies used
4. Describe UI elements and their functions in the workflow
5. Maintain chronological flow and logical sequence
6. Highlight important techniques or best practices demonstrated

The goal is to create a clear description of the entire workflow that someone could follow to understand or replicate the process.

VIDEO ANALYSIS:
{chunk}
"""

    _workflow_with_flowchart_prompt_template = """
You are analyzing a video that demonstrates workflow interactions with an AI platform. Below is a detailed analysis of the video frames.

Please create:
1. A well-structured, comprehensive summary of the AI interaction workflow shown in the video
2. A Mermaid flowchart diagram that visually represents the workflow

For the SUMMARY, include:
- Identification of distinct roles in the interaction (user types, AI roles)
- Description of the key steps in the workflow sequence
- Analysis of prompting patterns and strategies used
- Description of UI elements and their functions
- Logical sequence and dependencies between steps
- Important techniques or best practices demonstrated

For the FLOWCHART:
- Create a Mermaid flowchart diagram using the flowchart syntax
- Use clear, concise node labels
- Include all major steps in the process
- Show the relationships and flow between steps
- Include decision points where the workflow branches
- Group related steps if appropriate
- Keep the diagram clean and readable

First provide the summary, then create a section titled "Workflow Diagram" containing only the Mermaid code block for the flowchart.
Use this format for the Mermaid diagram:
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
    %% Add all needed nodes and connections
```

VIDEO ANALYSIS:
{chunk}
"""


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
        # For Ollama, create a proper Ollama model interface
        class OllamaModel:
            def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
                self.model_name = model_name
                self.base_url = base_url
                self.client = ollama.Client(host=base_url)
                logger.info(f"Initialized OllamaModel with model: {model_name}")

            def generate(self, prompt, **kwargs):
                # This method is no longer needed as we handle everything in __call__
                # Keeping it for backward compatibility
                logger.warning("OllamaModel.generate() is deprecated, use __call__() instead")
                try:
                    # Just delegate to __call__
                    return self.__call__(prompt, **kwargs)
                except Exception as e:
                    logger.error(f"Error in OllamaModel.generate: {str(e)}")
                    return f"Error calling Ollama API: {str(e)}"

            def __call__(self, messages, **kwargs):
                # This matches the interface expected by smolagents
                try:
                    # Format messages properly for Ollama
                    formatted_messages = []

                    # Handle different message formats
                    if isinstance(messages, str):
                        formatted_messages.append({"role": "user", "content": messages})
                    elif isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, str):
                                formatted_messages.append({"role": "user", "content": msg})
                            elif isinstance(msg, dict):
                                role = msg.get("role", "user")
                                content = msg.get("content", "")

                                # Handle content that is a list (like Anthropic format)
                                if isinstance(content, list):
                                    text_parts = []
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            text_parts.append(item["text"])
                                        elif isinstance(item, dict) and "type" in item and item["type"] == "text":
                                            text_parts.append(item.get("text", ""))

                                    formatted_messages.append({
                                        "role": role if role in ["user", "assistant", "system", "tool"] else "user",
                                        "content": " ".join(text_parts)
                                    })
                                else:
                                    # Ensure content is a string
                                    formatted_messages.append({
                                        "role": role if role in ["user", "assistant", "system", "tool"] else "user",
                                        "content": str(content)
                                    })
                            else:
                                logger.warning(f"Skipping invalid message format: {msg}")
                    else:
                        # Handle other types of input by converting to string
                        formatted_messages.append({"role": "user", "content": str(messages)})

                    # Extract relevant parameters
                    max_tokens = kwargs.get("max_tokens", 4096)
                    temperature = kwargs.get("temperature", 0.7)

                    # Call Ollama API directly
                    response = self.client.chat(
                        model=self.model_name,
                        messages=formatted_messages,
                        options={
                            "num_predict": max_tokens,
                            "temperature": temperature,
                            "stream": False
                        }
                    )

                    # Log the response type to help with debugging
                    logger.debug(f"Ollama response type: {type(response)}")

                    # Instead of trying to access attributes that might not exist,
                    # always convert to a string representation first

                    # Handle string response
                    if isinstance(response, str):
                        return response

                    # Handle dict response (common in newer Ollama versions)
                    if isinstance(response, dict):
                        # Try to get message.content
                        if 'message' in response:
                            message = response['message']
                            if isinstance(message, dict) and 'content' in message:
                                return str(message['content'])
                            # Handle other formats
                            return str(message)
                        # Try content directly
                        elif 'content' in response:
                            return str(response['content'])
                        # Try response field
                        elif 'response' in response:
                            return str(response['response'])

                    # Handle object with attributes (some Ollama client versions)
                    if hasattr(response, 'message'):
                        message = response.message
                        if hasattr(message, 'content'):
                            return str(message.content)
                        return str(message)

                    # Handle direct content attribute
                    if hasattr(response, 'content'):
                        return str(response.content)

                    # Handle direct response attribute
                    if hasattr(response, 'response'):
                        return str(response.response)

                    # Last resort: convert entire response to string
                    return str(response)

                except Exception as e:
                    logger.error(f"Error in OllamaModel.__call__: {str(e)}")
                    return f"Error calling Ollama API: {str(e)}"

            # Required methods for smolagents compatibility
            def get_num_tokens(self, text):
                # Simple approximation: 4 chars ~= 1 token
                return len(text) // 4

            def get_max_tokens(self):
                return 4096

        # Get model name from config
        ollama_model_name = model_config.ollama.model_name if hasattr(model_config.ollama, "model_name") else "llama3"
        ollama_base_url = model_config.ollama.base_url if hasattr(model_config.ollama, "base_url") else "http://localhost:11434"

        model = OllamaModel(model_name=ollama_model_name, base_url=ollama_base_url)
        logger.info(f"Using Ollama model: {ollama_model_name} for agent")
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

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")

    # Create output directory with datetime format
    output_dir = os.path.join("output", formatted_time)
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

    # Check if we need to use a smaller context window model for large videos
    # Claude-3-Opus has a 200k token limit
    try:
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return f"Error: Could not open video file: {video_path}"

        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        video_cap.release()

        # Estimate token count based on video duration and settings
        estimated_frames = duration / frame_interval
        if detect_scenes:
            # Scene detection typically reduces frame count by ~30-50%
            estimated_frames = estimated_frames * 0.7

        # Rough estimate: each frame analysis is ~1000 tokens
        estimated_tokens = estimated_frames * 1000

        # Calculate segment duration if needed
        segment_duration = end_time - start_time if end_time > start_time else duration

        # Adjust model and batch size based on estimated token count
        if estimated_tokens > 150000:  # Leave buffer below 200k limit
            logger.warning(f"Video is very long ({duration:.1f} seconds). Using Claude-3-Sonnet instead of Opus for better token efficiency.")
            if model_type == "anthropic" and vision_model_name == "claude":
                vision_model_name = "claude-3-sonnet-20240229"
                logger.info(f"Switched to {vision_model_name} for vision analysis")

            # Reduce batch size for very large videos
            if estimated_tokens > 180000:
                max_images_per_batch = min(max_images_per_batch, 8)
                logger.info(f"Reduced batch size to {max_images_per_batch} images per batch")

            # Recommend segmentation if not already segmented
            if segment_duration > 300 and segment_duration == duration:  # If processing more than 5 minutes at once
                logger.warning(f"Consider using --segment-duration 300 for better results with long videos")
    except Exception as e:
        logger.warning(f"Could not estimate video size: {str(e)}")

    # Run the agent to process the video
    task = f"""
Analyze the video at path "{video_path}" with the following steps:

1. Extract frames from the video using extract_frames tool
   - Use interval of {frame_interval} seconds
   - {'Detect scene changes' if detect_scenes else 'Do not detect scene changes'}
   - Use scene threshold of {scene_threshold}
   - Start at {start_time} seconds
   - {'End at ' + str(end_time) + ' seconds' if end_time > 0 else 'Process until the end of the video'}
   - Save all outputs to the directory: "{output_dir}"

2. {'Apply OCR to extract text from frames using extract_text_ocr tool' if enable_ocr else '# OCR disabled'}
   {'- Specify language as "' + language + '"' if enable_ocr else ''}
   {'- Save all outputs to the directory: "' + output_dir + '"' if enable_ocr else ''}

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
   - Save all outputs to the directory: "{output_dir}"

5. Generate a final coherent summary using generate_summary tool
   - Combine all the analyses
   - Use "{summary_model_name}" as the summary model
   {'- Use mission type "' + mission + '"' if mission != "general" else ''}
   {'- Generate a workflow flowchart' if generate_flowchart else ''}
   - Create a detailed, structured summary with clear sections for:
     * Overview/Introduction
     * Key Steps in the Workflow
     * Prompting Strategies
     * UI Elements and Functions
     * Logical Sequence and Dependencies
     * Best Practices Demonstrated
   - Provide granular details about each aspect of the workflow
   - Save all outputs to the directory: "{output_dir}"

The goal is to create a comprehensive analysis of the video, capturing all visual elements,
text content (with translations), and {'workflow logic' if mission == 'workflow' else 'narrative flow'}.
All output files should be saved to: "{output_dir}"
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
            "ollama_config": ollama_config.to_dict() if ollama_config.enabled else None,
            "output_dir": output_dir,  # Pass output directory to tools
            "max_batch_size_mb": max_batch_size_mb,
            "max_images_per_batch": max_images_per_batch,
            "batch_overlap_frames": batch_overlap_frames,
            "summary_sections": [
                "Overview",
                "Key Steps in the Workflow",
                "Prompting Strategies",
                "UI Elements and Functions",
                "Logical Sequence and Dependencies",
                "Best Practices Demonstrated"
            ],
            "summary_detail_level": "high",  # Request highly detailed summaries
            "include_examples": True,        # Include examples of prompts and responses
            "max_tokens_per_batch": 150000,  # Maximum tokens per batch for Claude
            "chunk_large_batches": True      # Enable chunking for large batches
        })

        # Log completion
        elapsed_time = time.time() - start_time_execution
        logger.info(f"Analysis completed in {elapsed_time:.1f} seconds")
        logger.info(f"Results saved to directory: {output_dir}")

        # Update result paths to include output directory
        if isinstance(result, dict):
            # Ensure all file paths in the result use the output directory
            for key in result:
                if isinstance(result[key], str) and (
                    key.endswith('_path') or key in ['coherent_summary', 'full_analysis', 'flowchart']
                ):
                    # If the path doesn't already include the output directory, update it
                    if not result[key].startswith(output_dir):
                        result[key] = os.path.join(output_dir, os.path.basename(result[key]))

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
    parser.add_argument("--segment-duration", type=float, default=0.0,
                        help="Process video in segments of this duration (in seconds, 0 = process entire video)")
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

    # Check if we need to process the video in segments
    if args.segment_duration > 0:
        try:
            # Get video duration
            video_cap = cv2.VideoCapture(args.video)
            if not video_cap.isOpened():
                print(f"Error: Could not open video file: {args.video}")
                return

            fps = video_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = frame_count / fps if fps > 0 else 0
            video_cap.release()

            # Create a timestamp for this run
            now = datetime.now()
            formatted_time = now.strftime("%Y%m%d%H%M")

            # Process video in segments
            segment_results = []
            start_time = args.start_time
            end_time = min(start_time + args.segment_duration, total_duration) if total_duration > 0 else start_time + args.segment_duration

            print(f"Processing video in segments of {args.segment_duration} seconds")
            segment_count = 1

            # Reduce batch size for segmented processing to avoid token limit issues
            video_config.max_images_per_batch = min(args.max_images_per_batch, 10)
            logger.info(f"Using reduced batch size of {video_config.max_images_per_batch} for segmented processing")

            while start_time < total_duration if total_duration > 0 else True:
                print(f"\nProcessing segment {segment_count}: {start_time:.1f}s to {end_time:.1f}s")

                # Update config for this segment
                video_config.start_time = start_time
                video_config.end_time = end_time

                # Run SmolaVision for this segment
                segment_result = run_smolavision(video_path=args.video, config=config)

                if isinstance(segment_result, dict) and "error" not in segment_result:
                    segment_results.append(segment_result)
                    print(f"Segment {segment_count} completed successfully")
                else:
                    print(f"Error processing segment {segment_count}: {segment_result}")

                # Move to next segment
                start_time = end_time
                end_time = min(start_time + args.segment_duration, total_duration) if total_duration > 0 else start_time + args.segment_duration
                segment_count += 1

                # Break if we've reached the end or if an end time was specified
                if (total_duration > 0 and start_time >= total_duration) or (args.end_time > 0 and start_time >= args.end_time):
                    break

            # Combine segment results into a single result
            if segment_results:
                # Create a combined result dictionary
                combined_result = segment_results[0].copy() if isinstance(segment_results[0], dict) else {}

                # Collect all analyses for summarization
                all_analyses = []
                for segment_result in segment_results:
                    if isinstance(segment_result, dict) and "analyses" in segment_result:
                        all_analyses.extend(segment_result["analyses"])

                # Generate a combined summary
                if all_analyses:
                    try:
                        # Create a new output directory for the combined result
                        combined_output_dir = os.path.join("output", f"{formatted_time}_combined")
                        os.makedirs(combined_output_dir, exist_ok=True)

                        # Create the summarization tool
                        summarization_tool = SummarizationTool()

                        # Generate the combined summary
                        summary_result = summarization_tool.forward(
                            analyses=all_analyses,
                            language=args.language,
                            model_name=model_config.summary_model,
                            api_key=api_key,
                            mission=args.mission,
                            generate_flowchart=args.generate_flowchart,
                            output_dir=combined_output_dir
                        )

                        # Update the combined result with the summary
                        if isinstance(summary_result, dict):
                            for key, value in summary_result.items():
                                combined_result[key] = value

                        # Update paths to use the combined output directory
                        for key in combined_result:
                            if isinstance(combined_result[key], str) and (
                                key.endswith('_path') or key in ['coherent_summary', 'full_analysis', 'flowchart']
                            ):
                                # Update the path to use the combined output directory
                                combined_result[key] = os.path.join(
                                    combined_output_dir,
                                    os.path.basename(combined_result[key])
                                )

                        result = combined_result
                        logger.info(f"Combined summary generated in: {combined_output_dir}")
                    except Exception as e:
                        logger.error(f"Error generating combined summary: {str(e)}")
                        # Fall back to the last segment result
                        result = segment_results[-1]
                        logger.info("Using last segment result as fallback")
                else:
                    # Fall back to the last segment result
                    result = segment_results[-1]
                    logger.info("No analyses found, using last segment result")
            else:
                result = "No segments processed successfully"
        except Exception as e:
            print(f"Error processing video in segments: {str(e)}")
            result = f"Error: {str(e)}"
    else:
        # Run SmolaVision for the entire video
        result = run_smolavision(video_path=args.video, config=config)

    # Print result information
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSummary of video:")
            print("-" * 80)

            # Print a more structured summary with sections
            summary_text = result.get("summary_text", "")

            # Try to extract sections from the summary if they exist
            sections = {}
            current_section = "Overview"
            section_content = []

            for line in summary_text.split('\n'):
                if line.strip() and (line.strip().endswith(':') or
                                    line.strip().startswith('# ') or
                                    line.strip().startswith('## ')):
                    # This looks like a section header
                    if section_content:
                        sections[current_section] = '\n'.join(section_content)
                        section_content = []
                    current_section = line.strip().replace('# ', '').replace('## ', '')
                    if current_section.endswith(':'):
                        current_section = current_section[:-1]
                else:
                    section_content.append(line)

            # Add the last section
            if section_content:
                sections[current_section] = '\n'.join(section_content)

            # Print the structured summary
            if sections and len(sections) > 1:
                for section, content in sections.items():
                    print(f"\n{section}:")
                    print("-" * len(section) + "-")
                    print(content.strip())
            else:
                # Fall back to the original summary if no sections were found
                print(summary_text[:1000] + "..." if len(summary_text) > 1000 else summary_text)

            print("\n" + "-" * 80)
            print(f"Full summary saved to: {result['coherent_summary']}")
            print(f"Full analysis saved to: {result['full_analysis']}")

            if "flowchart" in result:
                print(f"Workflow flowchart saved to: {result['flowchart']}")

            # Provide a hint about viewing the mermaid diagram
            if "flowchart" in result and result["flowchart"].endswith(".mmd"):
                print("\nTo view the flowchart, you can use the Mermaid Live Editor:")
                print("1. Open https://mermaid.live/")
                print("2. Copy the contents of the .mmd file and paste it into the editor")
    else:
        print(result)


if __name__ == "__main__":
    main()
