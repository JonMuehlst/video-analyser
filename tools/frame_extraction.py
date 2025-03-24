#!/usr/bin/env python
"""
Frame extraction tool for SmolaVision
"""

import os
import cv2
import base64
import gc
import logging
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
from io import BytesIO

from smolagents import Tool

# Configure logging
logger = logging.getLogger("SmolaVision")


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
