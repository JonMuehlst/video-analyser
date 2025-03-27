# smolavision/video/extractor.py
import cv2
import base64
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import timedelta
from PIL import Image
from smolavision.exceptions import VideoProcessingError
from smolavision.video.types import Frame
from smolavision.video.base import FrameExtractor

logger = logging.getLogger(__name__)

class DefaultFrameExtractor(FrameExtractor):
    """Default implementation of the FrameExtractor interface."""
    
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
            List of extracted frames
            
        Raises:
            VideoProcessingError: If video processing fails
        """
        return extract_frames(
            video_path=video_path,
            interval_seconds=interval_seconds,
            detect_scenes=detect_scenes,
            scene_threshold=scene_threshold,
            resize_width=resize_width,
            start_time=start_time,
            end_time=end_time
        )

# Function for backward compatibility and direct usage
def extract_frames(
    video_path: str,
    interval_seconds: int = 10,
    detect_scenes: bool = True,
    scene_threshold: float = 30.0,
    resize_width: Optional[int] = None,
    start_time: float = 0.0,
    end_time: float = 0.0
) -> List[Frame]:
    """
    Extract frames from a video file at regular intervals and detect scene changes.

    Args:
        video_path: Path to the video file
        interval_seconds: Extract a frame every N seconds
        detect_scenes: Whether to detect scene changes
        scene_threshold: Threshold for scene change detection (higher = less sensitive)
        resize_width: Width to resize frames to (keeps aspect ratio)
        start_time: Start time in seconds (0 for beginning)
        end_time: End time in seconds (0 for entire video)

    Returns:
        List of extracted frames with metadata

    Raises:
        VideoProcessingError: If video processing fails
    """
    logger.info(f"Extracting frames from {video_path}")
    extracted_frames: List[Frame] = []  # Explicitly type it

    try:
        # Open video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise VideoProcessingError(f"Could not open video: {video_path}")

        frame_rate = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_seconds = total_frames / frame_rate

        start_frame = int(start_time * frame_rate) if start_time > 0 else 0
        end_frame = int(end_time * frame_rate) if end_time > 0 else total_frames

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set start position

        frame_number = start_frame  # Keep track of the actual frame number
        while frame_number < end_frame:
            success, image = video.read()
            if not success:
                break

            timestamp = frame_number / frame_rate

            # Resize image if requested (before encoding)
            if resize_width is not None:
                height, width = image.shape[:2]
                aspect_ratio = width / height
                new_height = int(resize_width / aspect_ratio)
                image = cv2.resize(image, (resize_width, new_height))

            # Convert image to base64
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")  # JPEG is more efficient
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Decode bytes to string


            extracted_frames.append(
                Frame(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    image_data=img_str,  # Store base64 encoded data
                    scene_change=False, # Placeholder. Scene detection will be implemented later
                    metadata={}  # Placeholder.  Add resolution etc. later
                )
            )

            frame_number += int(interval_seconds * frame_rate)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # fast forward video to next capture

        video.release() # close video
        logger.info(f"Extracted {len(extracted_frames)} frames.")
        return extracted_frames

    except cv2.error as e:
        raise VideoProcessingError(f"OpenCV error: {str(e)}") from e
    except Exception as e:
        raise VideoProcessingError(f"Unexpected error: {str(e)}") from e
