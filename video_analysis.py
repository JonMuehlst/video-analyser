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
from datetime import timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from collections import deque
from PIL import Image
from io import BytesIO

# Import smolagents
from smolagents import CodeAgent, Tool, HfApiModel, LiteLLMModel

# Import local modules
from config import create_default_config, ModelConfig, VideoConfig
from ollama_client import OllamaClient
from utils import ensure_directory, format_time_seconds

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
                # Use Ollama for local inference
                base_url = ollama_config.get("base_url", "http://localhost:11434")
                vision_model = ollama_config.get("vision_model", "llava")
                
                # Create or reuse Ollama client
                if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                    self._ollama_client = OllamaClient(base_url=base_url)
                
                # Extract base64 images for Ollama
                images = []
                for frame in batch:
                    base64_image = frame.get('base64_image', '')
                    if base64_image and isinstance(base64_image, str):
                        images.append(base64_image)
                
                # Call Ollama vision model
                analysis = self._ollama_client.generate_vision(
                    model=vision_model,
                    prompt=prompt,
                    images=images,
                    max_tokens=4096
                )
                
            # Prepare images based on the model
            elif model_name.startswith("claude"):
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
                response = self._anthropic_client.messages.create(
                    model=model_id,
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
                analysis = response.content[0].text

            elif model_name == "gpt4o" or model_name == "gpt-4o":
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
                    # Use Ollama for local inference
                    base_url = ollama_config.get("base_url", "http://localhost:11434")
                    model = ollama_config.get("model_name", "llama3")
                    
                    # Create or reuse Ollama client
                    if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                        self._ollama_client = OllamaClient(base_url=base_url)
                    
                    # Call Ollama model
                    chunk_summary = self._ollama_client.generate(
                        model=model,
                        prompt=prompt,
                        max_tokens=4096
                    )
                
                # Call the LLM API based on model name
                elif model_name.startswith("claude"):
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

                elif model_name.startswith("gpt"):
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
    model_config = config.get("model", {})
    api_key = model_config.get("api_key", "")
    model_type = model_config.get("model_type", "anthropic")

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
        model_config["api_key"] = api_key
    if model_type is not None:
        model_config["model_type"] = model_type
    if vision_model is not None:
        model_config["vision_model"] = vision_model
    if summary_model is not None:
        model_config["summary_model"] = summary_model
    
    # Update Ollama configuration
    ollama_config = model_config.get("ollama", {})
    if ollama_enabled is not None:
        ollama_config["enabled"] = ollama_enabled
    if ollama_base_url is not None:
        ollama_config["base_url"] = ollama_base_url
    if ollama_model is not None:
        ollama_config["model_name"] = ollama_model
    if ollama_vision_model is not None:
        ollama_config["vision_model"] = ollama_vision_model
    model_config["ollama"] = ollama_config
    
    # Update video configuration
    if language is not None:
        video_config["language"] = language
    if frame_interval is not None:
        video_config["frame_interval"] = frame_interval
    if detect_scenes is not None:
        video_config["detect_scenes"] = detect_scenes
    if scene_threshold is not None:
        video_config["scene_threshold"] = scene_threshold
    if enable_ocr is not None:
        video_config["enable_ocr"] = enable_ocr
    if start_time is not None:
        video_config["start_time"] = start_time
    if end_time is not None:
        video_config["end_time"] = end_time
    if mission is not None:
        video_config["mission"] = mission
    if generate_flowchart is not None:
        video_config["generate_flowchart"] = generate_flowchart
    
    # Update the config with modified sections
    config["model"] = model_config
    config["video"] = video_config
    
    # Extract configuration values for use in the task
    api_key = model_config.get("api_key", "")
    model_type = model_config.get("model_type", "anthropic")
    vision_model = model_config.get("vision_model", "claude")
    summary_model = model_config.get("summary_model", "claude-3-5-sonnet-20240620")
    ollama_config = model_config.get("ollama", {})
    
    language = video_config.get("language", "Hebrew")
    frame_interval = video_config.get("frame_interval", 10)
    detect_scenes = video_config.get("detect_scenes", True)
    scene_threshold = video_config.get("scene_threshold", 30.0)
    enable_ocr = video_config.get("enable_ocr", True)
    start_time = video_config.get("start_time", 0.0)
    end_time = video_config.get("end_time", 0.0)
    mission = video_config.get("mission", "general")
    generate_flowchart = video_config.get("generate_flowchart", False)
    max_batch_size_mb = video_config.get("max_batch_size_mb", 10.0)
    max_images_per_batch = video_config.get("max_images_per_batch", 15)
    batch_overlap_frames = video_config.get("batch_overlap_frames", 2)

    # Create the agent
    agent = create_smolavision_agent(config)

    # Determine which model to use for vision analysis
    vision_model_name = vision_model
    if ollama_config.get("enabled"):
        vision_model_name = "ollama"
        logger.info(f"Using Ollama for vision analysis with model: {ollama_config.get('vision_model')}")

    # Determine which model to use for summarization
    summary_model_name = summary_model
    if ollama_config.get("enabled"):
        summary_model_name = "ollama"
        logger.info(f"Using Ollama for summarization with model: {ollama_config.get('model_name')}")

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
            "ollama_config": ollama_config if ollama_config.get("enabled") else None
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
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Create configuration
    config = create_default_config(api_key)
    
    # Update model configuration
    config["model"]["model_type"] = args.model_type
    config["model"]["vision_model"] = args.vision_model
    config["model"]["summary_model"] = args.summary_model
    
    # Update Ollama configuration
    config["model"]["ollama"]["enabled"] = args.ollama_enabled or args.model_type == "ollama"
    config["model"]["ollama"]["base_url"] = args.ollama_base_url
    config["model"]["ollama"]["model_name"] = args.ollama_model
    config["model"]["ollama"]["vision_model"] = args.ollama_vision_model
    
    # Update video configuration
    config["video"]["language"] = args.language
    config["video"]["frame_interval"] = args.frame_interval
    config["video"]["detect_scenes"] = args.detect_scenes
    config["video"]["scene_threshold"] = args.scene_threshold
    config["video"]["enable_ocr"] = args.enable_ocr
    config["video"]["start_time"] = args.start_time
    config["video"]["end_time"] = args.end_time
    config["video"]["mission"] = args.mission
    config["video"]["generate_flowchart"] = args.generate_flowchart
    config["video"]["max_batch_size_mb"] = args.max_batch_size_mb
    config["video"]["max_images_per_batch"] = args.max_images_per_batch
    config["video"]["batch_overlap_frames"] = args.batch_overlap_frames

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
