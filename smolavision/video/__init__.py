# smolavision/video/__init__.py
from smolavision.video.base import FrameExtractor
from smolavision.video.extractor import extract_frames, DefaultFrameExtractor
from smolavision.video.scene_detection import detect_scene_changes
from smolavision.video.utils import get_video_resolution
from smolavision.video.types import Frame

__all__ = [
    "FrameExtractor",
    "DefaultFrameExtractor",
    "extract_frames",
    "detect_scene_changes",
    "get_video_resolution",
    "Frame"
]
