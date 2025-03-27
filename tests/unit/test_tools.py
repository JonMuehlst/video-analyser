import pytest
from unittest.mock import patch, MagicMock

from smolavision.tools.frame_extraction import FrameExtractionTool
from smolavision.tools.ocr_extraction import OCRExtractionTool
from smolavision.tools.batch_creation import BatchCreationTool
from smolavision.tools.vision_analysis import VisionAnalysisTool
from smolavision.tools.summarization import SummarizationTool
from smolavision.tools.factory import ToolFactory
from smolavision.exceptions import ToolError, ConfigurationError


class TestFrameExtractionTool:
    """Tests for the frame extraction tool."""

    @patch('smolavision.video.extractor.extract_frames')
    def test_use(self, mock_extract_frames):
        """Test using the frame extraction tool."""
        # Mock extract_frames to return a list of frames
        mock_frames = [
            {"frame_number": 0, "timestamp": 0.0, "image_data": "base64data", "scene_change": False, "metadata": {}}
        ]
        mock_extract_frames.return_value = mock_frames
        
        # Create tool and call use
        config = {"video": {"frame_interval": 10}}
        tool = FrameExtractionTool(config)
        result = tool.use("test_video.mp4")
        
        # Verify results
        assert result == str(mock_frames)
        mock_extract_frames.assert_called_once_with(
            video_path="test_video.mp4",
            interval_seconds=10,
            detect_scenes=True,
            scene_threshold=30.0,
            resize_width=None,
            start_time=0.0,
            end_time=0.0
        )

    @patch('smolavision.video.extractor.extract_frames', side_effect=Exception("Extraction error"))
    def test_error_handling(self, mock_extract_frames):
        """Test error handling in the frame extraction tool."""
        # Create tool and call use
        config = {"video": {}}
        tool = FrameExtractionTool(config)
        
        # Verify exception is raised
        with pytest.raises(ToolError):
            tool.use("test_video.mp4")


class TestOCRExtractionTool:
    """Tests for the OCR extraction tool."""

    @patch('smolavision.ocr.extractor.extract_text')
    def test_use(self, mock_extract_text):
        """Test using the OCR extraction tool."""
        # Mock extract_text to return a list of frames
        mock_frames = [
            {"frame_number": 0, "timestamp": 0.0, "image_data": "base64data", "scene_change": False, "ocr_text": "Text", "metadata": {}}
        ]
        mock_extract_text.return_value = mock_frames
        
        # Create tool and call use
        config = {"video": {"language": "English"}}
        tool = OCRExtractionTool(config)
        result = tool.use([{"frame_number": 0, "timestamp": 0.0, "image_data": "base64data", "scene_change": False, "metadata": {}}])
        
        # Verify results
        assert result == str(mock_frames)
        mock_extract_text.assert_called_once()


class TestBatchCreationTool:
    """Tests for the batch creation tool."""

    @patch('smolavision.batch.creator.create_batches')
    def test_use(self, mock_create_batches):
        """Test using the batch creation tool."""
        # Mock create_batches to return a list of batches
        mock_batches = [
            {"frames": [0], "image_data": ["base64data"], "timestamps": [0.0], "ocr_text": ["Text"]}
        ]
        mock_create_batches.return_value = mock_batches
        
        # Create tool and call use
        config = {"analysis": {"max_batch_size_mb": 10.0}}
        tool = BatchCreationTool(config)
        result = tool.use([{"frame_number": 0, "timestamp": 0.0, "image_data": "base64data", "scene_change": False, "metadata": {}}])
        
        # Verify results
        assert result == str(mock_batches)
        mock_create_batches.assert_called_once()


class TestVisionAnalysisTool:
    """Tests for the vision analysis tool."""

    @patch('smolavision.models.factory.create_vision_model')
    def test_use(self, mock_create_model):
        """Test using the vision analysis tool."""
        # Mock model
        mock_model = MagicMock()
        mock_model.analyze_images.return_value = "Analysis result"
        mock_create_model.return_value = mock_model
        
        # Create tool and call use
        config = {"model": {}}
        tool = VisionAnalysisTool(config)
        result = tool.use(
            [{"frames": [0], "image_data": ["base64data"], "timestamps": [0.0], "ocr_text": ["Text"]}],
            prompt="Analyze these images"
        )
        
        # Verify results
        assert result == "Analysis result"
        mock_model.analyze_images.assert_called_once()


class TestSummarizationTool:
    """Tests for the summarization tool."""

    @patch('smolavision.models.factory.create_summary_model')
    def test_use(self, mock_create_model):
        """Test using the summarization tool."""
        # Mock model
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "Summary result"
        mock_create_model.return_value = mock_model
        
        # Create tool and call use
        config = {"model": {}}
        tool = SummarizationTool(config)
        result = tool.use(["Analysis 1", "Analysis 2"], language="English")
        
        # Verify results
        assert result == "Summary result"
        mock_model.generate_text.assert_called_once()


class TestToolFactory:
    """Tests for the tool factory."""

    def test_create_tool(self):
        """Test creating tools with the factory."""
        config = {}
        
        # Test creating each tool type
        frame_tool = ToolFactory.create_tool(config, "frame_extraction")
        assert isinstance(frame_tool, FrameExtractionTool)
        
        ocr_tool = ToolFactory.create_tool(config, "ocr_extraction")
        assert isinstance(ocr_tool, OCRExtractionTool)
        
        batch_tool = ToolFactory.create_tool(config, "batch_creation")
        assert isinstance(batch_tool, BatchCreationTool)
        
        vision_tool = ToolFactory.create_tool(config, "vision_analysis")
        assert isinstance(vision_tool, VisionAnalysisTool)
        
        summary_tool = ToolFactory.create_tool(config, "summarization")
        assert isinstance(summary_tool, SummarizationTool)

    def test_invalid_tool_type(self):
        """Test error handling for invalid tool type."""
        config = {}
        
        with pytest.raises(ConfigurationError):
            ToolFactory.create_tool(config, "invalid_tool")

    def test_register_tool(self):
        """Test registering a custom tool."""
        # Create a mock tool class
        mock_tool_class = MagicMock()
        
        # Register the tool
        ToolFactory.register_tool("custom_tool", mock_tool_class)
        
        # Create the tool
        config = {}
        ToolFactory.create_tool(config, "custom_tool")
        
        # Verify the tool was created
        mock_tool_class.assert_called_once_with(config)
