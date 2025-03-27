# smolavision/video/utils.py
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get the resolution of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        A tuple containing the width and height of the video.
    """
    import cv2

    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()
        return width, height
    except cv2.error as e:
        logger.error(f"OpenCV error: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return None, None