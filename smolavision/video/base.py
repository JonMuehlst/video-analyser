from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from smolavision.video.types import Frame

class FrameExtractor(ABC):
    """Base interface for video frame extractors."""
    
    @abstractmethod
    def extract_frames(
        self,
        video_path: str,
        interval_seconds: int = 10,
        detect_scenes: bool = True,
        scene_threshold: float = 30.0,
        resize_width: Optional[int] = None,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> List[Frame]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Extract a frame every N seconds
            detect_scenes: Whether to detect scene changes
            scene_threshold: Threshold for scene change detection
            resize_width: Width to resize frames to (keeps aspect ratio)
            start_time: Start time in seconds (0 for beginning)
            end_time: End time in seconds (0 for entire video)
            
        Returns:
            List of extracted frames
        """
        pass
