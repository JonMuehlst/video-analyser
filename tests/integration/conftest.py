import os
import pytest
import tempfile
import shutil
from typing import Dict, Any

from smolavision.config.schema import Config
from smolavision.models.factory import ModelFactory
from smolavision.pipeline.factory import create_pipeline


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = Config().to_dict()
    config["output_dir"] = "test_output"
    config["model"]["model_type"] = "ollama"  # Use Ollama for tests as it doesn't require API keys
    config["model"]["ollama"]["enabled"] = True
    config["video"]["frame_interval"] = 30  # Extract fewer frames for faster tests
    config["video"]["enable_ocr"] = False  # Disable OCR for faster tests
    config["analysis"]["max_images_per_batch"] = 5  # Smaller batches for faster tests
    return config


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    from unittest.mock import MagicMock
    
    # Create mock models
    vision_model = MagicMock()
    vision_model.analyze_images.return_value = "Vision analysis result"
    
    summary_model = MagicMock()
    summary_model.generate_text.return_value = "Summary result"
    
    return {
        "vision": vision_model,
        "summary": summary_model
    }


@pytest.fixture
def test_video_path():
    """Return path to a test video file."""
    # This is a placeholder - in a real test you'd use a real test video
    # For CI/CD, you might want to generate a synthetic video
    video_path = os.path.join(os.path.dirname(__file__), "resources", "test_video.mp4")
    
    # If the test video doesn't exist, create a simple one
    if not os.path.exists(video_path):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Create a simple video using OpenCV
        import cv2
        import numpy as np
        
        # Create a 5-second video with a white background
        fps = 30
        width, height = 640, 480
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for i in range(5 * fps):
            # Create a white frame
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add a counter
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
    
    return video_path
