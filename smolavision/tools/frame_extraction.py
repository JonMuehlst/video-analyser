"""
Frame extraction tool adapter for SmolaVision.
"""

import logging
from typing import List, Dict, Any, Optional

from ..exceptions import VideoProcessingError
from ..logging import get_logger
from ..video.extractor import extract_frames

logger = get_logger("tools.frame_extraction")


class FrameExtractionTool:
    """Adapter for frame extraction functionality"""
    
    @staticmethod
    def execute(
        video_path: str,
        interval_seconds: int = 10,
        detect_scenes: bool = True,
        scene_threshold: float = 30.0,
        resize_width: Optional[int] = None,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from a video file at regular intervals and detect scene changes.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Extract a frame every N seconds
            detect_scenes: Whether to detect scene changes
            scene_threshold: Threshold for scene change detection
            resize_width: Width to resize frames to (keeps aspect ratio)
            start_time: Start time in seconds (0 for beginning)
            end_time: End time in seconds (0 for entire video)
            
        Returns:
            List of dictionaries containing frame data
            
        Raises:
            VideoProcessingError: If frame extraction fails
        """
        try:
            return extract_frames(
                video_path=video_path,
                interval_seconds=interval_seconds,
                resize_width=resize_width,
                start_time=start_time,
                end_time=end_time,
                detect_scenes=detect_scenes,
                scene_threshold=scene_threshold
            )
        except Exception as e:
            error_msg = f"Error extracting frames: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e
