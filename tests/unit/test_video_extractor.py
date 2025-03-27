import pytest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from smolavision.video.extractor import extract_frames, DefaultFrameExtractor
from smolavision.exceptions import VideoProcessingError


class TestVideoExtractor:
    """Tests for the video extraction functionality."""

    def test_extract_frames_invalid_path(self):
        """Test that extract_frames raises an error for invalid video path."""
        with pytest.raises(VideoProcessingError):
            extract_frames("nonexistent_video.mp4")

    @patch('cv2.VideoCapture')
    def test_extract_frames_basic(self, mock_video_capture):
        """Test basic frame extraction functionality."""
        # Mock VideoCapture
        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
        }.get(prop, 0)
        
        # Mock read to return one frame then False
        mock_video.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_video_capture.return_value = mock_video

        # Call the function
        frames = extract_frames("test_video.mp4", interval_seconds=10)
        
        # Verify results
        assert len(frames) == 1
        assert frames[0].frame_number == 0
        assert frames[0].timestamp == 0.0
        assert isinstance(frames[0].image_data, str)  # Should be base64 encoded

    @patch('cv2.VideoCapture')
    def test_extract_frames_with_start_end_time(self, mock_video_capture):
        """Test frame extraction with start and end times."""
        # Mock VideoCapture
        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        
        # Mock read to return frames
        mock_video.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_video

        # Call the function with start and end times
        frames = extract_frames(
            "test_video.mp4", 
            interval_seconds=5,
            start_time=10.0,
            end_time=20.0
        )
        
        # Verify results - should extract frames at 10s, 15s
        assert len(frames) == 2
        assert frames[0].timestamp >= 10.0
        assert frames[1].timestamp <= 20.0

    def test_default_frame_extractor(self):
        """Test the DefaultFrameExtractor class."""
        extractor = DefaultFrameExtractor()
        
        # Patch the extract_frames function that DefaultFrameExtractor uses
        with patch('smolavision.video.extractor.extract_frames') as mock_extract:
            mock_extract.return_value = [MagicMock()]
            
            # Call the extractor
            result = extractor.extract_frames("test_video.mp4")
            
            # Verify the function was called with the right parameters
            mock_extract.assert_called_once_with(
                video_path="test_video.mp4",
                interval_seconds=10,
                detect_scenes=True,
                scene_threshold=30.0,
                resize_width=None,
                start_time=0.0,
                end_time=0.0
            )
            
            assert len(result) == 1
