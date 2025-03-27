import os
import logging
from typing import Dict, Any, Optional

from smolavision.config import create_default_config
from smolavision.pipeline.factory import create_pipeline

logger = logging.getLogger(__name__)

def run_smolavision(
    video_path: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the SmolaVision pipeline on a video.
    
    Args:
        video_path: Path to the video file
        config: Configuration dictionary (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with analysis results
    """
    # Create default configuration if none provided
    if config is None:
        config = create_default_config()
    
    # Override config with any explicitly provided parameters
    for section in ["video", "model", "analysis"]:
        if section in config:
            for key, value in kwargs.items():
                if key in config[section] and value is not None:
                    config[section][key] = value
    
    # Create and run pipeline
    pipeline = create_pipeline(config)
    return pipeline.run(video_path)
