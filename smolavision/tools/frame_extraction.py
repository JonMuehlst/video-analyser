# smolavision/tools/frame_extraction.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.video.extractor import extract_frames
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class FrameExtractionTool(Tool):
    """Tool for extracting frames from a video."""

    name: str = "frame_extraction"
    description: str = "Extract frames from a video at specified intervals. Specify path to video."
    input_type: str = "video_path"
    output_type: str = "List[Dict[str, Any]]" # Output is now a list of dicts

    def __init__(self, config: Dict[str, Any]):
        """Initialize the FrameExtractionTool."""
        self.config = config.get("video", {})  # Get nested "video" part of config

    def use(self, video_path: str) -> str:
        """
        Extract frames from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            String representation of the extracted frames
            
        Raises:
            ToolError: If frame extraction fails
        """
        try:
            frames = extract_frames(
                video_path=video_path,
                interval_seconds=self.config.get("frame_interval", 10),
                detect_scenes=self.config.get("detect_scenes", True),
                scene_threshold=self.config.get("scene_threshold", 30.0),
                resize_width=self.config.get("resize_width"),
                start_time=self.config.get("start_time", 0.0),
                end_time=self.config.get("end_time", 0.0)
            )
            # Return the actual list of frame dictionaries
            return [frame.model_dump() for frame in frames]

        except Exception as e:
            raise ToolError(f"Frame extraction failed: {e}") from e
