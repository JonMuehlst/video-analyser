# smolavision/tools/batch_creation.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.video.types import Frame
from smolavision.batch.creator import create_batches
from smolavision.batch.types import Batch
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class BatchCreationTool(Tool):
    """Tool for creating batches of frames for efficient analysis."""

    name: str = "batch_creation"
    description: str = "Create batches of frames for analysis, provide a list of frames to batch"
    input_type: str = "list[Frame]"
    output_type: str = "list[Batch]"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BatchCreationTool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("analysis", {})
        logger.debug(f"BatchCreationTool initialized with config: {self.config}")

    def use(self, frames: List[Dict[str, Any]]) -> str:
        """
        Create batches from the input frames.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            String representation of created batches
            
        Raises:
            ToolError: If batch creation fails
        """
        try:
            # Convert dictionaries to Frame objects
            typed_frames = [Frame(**frame) for frame in frames]
            
            # Get batch parameters from config
            max_batch_size_mb = self.config.get("max_batch_size_mb", 10.0)
            max_images_per_batch = self.config.get("max_images_per_batch", 15)
            overlap_frames = self.config.get("batch_overlap_frames", 2)
            
            logger.info(f"Creating batches from {len(typed_frames)} frames "
                       f"(max_size: {max_batch_size_mb}MB, "
                       f"max_images: {max_images_per_batch}, "
                       f"overlap: {overlap_frames})")
            
            # Create batches
            batches = create_batches(
                frames=typed_frames,
                max_batch_size_mb=max_batch_size_mb,
                max_images_per_batch=max_images_per_batch,
                overlap_frames=overlap_frames
            )
            
            logger.info(f"Created {len(batches)} batches")
            
            # Convert batches to dictionaries and return as string
            return str([batch.model_dump() for batch in batches])
            
        except Exception as e:
            logger.exception("Error during batch creation")
            raise ToolError(f"Batch creation failed: {e}") from e
