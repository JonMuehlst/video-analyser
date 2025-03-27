from abc import ABC, abstractmethod
from typing import Dict, Any

class Pipeline(ABC):
    """Base interface for video analysis pipelines."""
    
    @abstractmethod
    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Run the pipeline on a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with analysis results
        """
        pass
