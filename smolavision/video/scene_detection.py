# smolavision/video/scene_detection.py
import logging
import base64
import cv2
import numpy as np
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from smolavision.video.types import Frame

logger = logging.getLogger(__name__)

def detect_scene_changes(frames: List[Frame], threshold: float = 30.0) -> List[Frame]:
    """
    Detect scene changes in a list of frames.

    Args:
        frames: List of Frame objects
        threshold: Threshold for scene change detection (higher = less sensitive)

    Returns:
        List of Frame objects with the 'scene_change' attribute set to True
        for frames where a scene change is detected.
    """
    logger.info(f"Detecting scene changes with threshold {threshold}...")
    
    if not frames:
        return frames
    
    # Mark the first frame as a scene change
    frames[0].scene_change = True
    
    # We need at least 2 frames to detect changes
    if len(frames) < 2:
        return frames
    
    prev_hist = None
    
    for i, frame in enumerate(frames):
        # Skip the first frame as we already marked it
        if i == 0:
            continue
        
        # Convert base64 to image
        try:
            img_data = base64.b64decode(frame.image_data)
            img = np.array(Image.open(BytesIO(img_data)))
            
            # Convert to grayscale for histogram comparison
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                # Compare histograms
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) * 100
                
                # If difference exceeds threshold, mark as scene change
                if diff > threshold:
                    frame.scene_change = True
                    logger.debug(f"Scene change detected at frame {frame.frame_number} (diff: {diff:.2f})")
                else:
                    frame.scene_change = False
            
            # Update previous histogram
            prev_hist = hist
            
        except Exception as e:
            logger.warning(f"Error processing frame {frame.frame_number} for scene detection: {e}")
            frame.scene_change = False
    
    # Count scene changes
    scene_changes = sum(1 for frame in frames if frame.scene_change)
    logger.info(f"Detected {scene_changes} scene changes in {len(frames)} frames")
    
    return frames
