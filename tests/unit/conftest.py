import os
import pytest
import base64
from typing import List, Dict, Any
from unittest.mock import MagicMock

from smolavision.video.types import Frame
from smolavision.batch.types import Batch
from smolavision.models.base import ModelInterface


@pytest.fixture
def sample_image_data() -> str:
    """Return a small base64 encoded test image."""
    # 1x1 pixel transparent PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


@pytest.fixture
def sample_frames(sample_image_data) -> List[Frame]:
    """Return a list of sample frames for testing."""
    return [
        Frame(
            frame_number=0,
            timestamp=0.0,
            image_data=sample_image_data,
            scene_change=True,
            metadata={}
        ),
        Frame(
            frame_number=10,
            timestamp=10.0,
            image_data=sample_image_data,
            scene_change=False,
            metadata={}
        ),
        Frame(
            frame_number=20,
            timestamp=20.0,
            image_data=sample_image_data,
            scene_change=True,
            metadata={}
        ),
    ]


@pytest.fixture
def sample_batch(sample_frames) -> Batch:
    """Return a sample batch for testing."""
    return Batch(
        frames=[frame.frame_number for frame in sample_frames],
        timestamps=[frame.timestamp for frame in sample_frames],
        image_data=[frame.image_data for frame in sample_frames],
        ocr_text=["Sample text 1", "Sample text 2", "Sample text 3"]
    )


@pytest.fixture
def mock_model() -> ModelInterface:
    """Return a mock model for testing."""
    mock = MagicMock(spec=ModelInterface)
    mock.generate_text.return_value = "Generated text response"
    mock.analyze_images.return_value = "Image analysis response"
    return mock


@pytest.fixture
def test_video_path() -> str:
    """Return path to a test video file."""
    # This is a placeholder - in a real test you'd use a real test video
    return os.path.join(os.path.dirname(__file__), "resources", "test_video.mp4")
