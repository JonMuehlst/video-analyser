"""
Batch creation tool adapter for SmolaVision.
"""

import logging
from typing import List, Dict, Any

from ..exceptions import BatchProcessingError
from ..logging import get_logger
from ..batch.creator import create_batches

logger = get_logger("tools.batch_creation")


class BatchCreationTool:
    """Adapter for batch creation functionality"""
    
    @staticmethod
    def execute(
        frames: List[Dict[str, Any]],
        max_batch_size_mb: float = 10.0,
        max_images_per_batch: int = 15,
        overlap_frames: int = 2
    ) -> List[List[Dict[str, Any]]]:
        """
        Group frames into batches based on size and count limits.
        
        Args:
            frames: List of extracted frames
            max_batch_size_mb: Maximum size of a batch in MB
            max_images_per_batch: Maximum number of images in a batch
            overlap_frames: Number of frames to overlap between batches
            
        Returns:
            List of batches, where each batch is a list of frames
            
        Raises:
            BatchProcessingError: If batch creation fails
        """
        try:
            return create_batches(
                frames=frames,
                max_batch_size_mb=max_batch_size_mb,
                max_images_per_batch=max_images_per_batch,
                overlap_frames=overlap_frames
            )
        except Exception as e:
            error_msg = f"Error creating batches: {str(e)}"
            logger.error(error_msg)
            raise BatchProcessingError(error_msg) from e
