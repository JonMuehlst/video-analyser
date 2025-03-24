#!/usr/bin/env python
"""
Batch creation tool for SmolaVision
"""

import logging
from typing import List, Dict, Any

from smolagents import Tool

# Configure logging
logger = logging.getLogger("SmolaVision")


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
