import pytest
import os
from unittest.mock import patch

from smolavision.pipeline.standard import StandardPipeline
from smolavision.pipeline.segmented import SegmentedPipeline
from smolavision.exceptions import PipelineError


class TestStandardPipeline:
    """Integration tests for the standard pipeline."""

    @patch('smolavision.video.extractor.extract_frames')
    @patch('smolavision.batch.creator.create_batches')
    @patch('smolavision.analysis.vision.analyze_batch')
    @patch('smolavision.analysis.summarization.generate_summary')
    def test_pipeline_execution(self, mock_generate_summary, mock_analyze_batch, 
                               mock_create_batches, mock_extract_frames, 
                               test_config, mock_models, temp_output_dir):
        """Test the complete execution of the standard pipeline."""
        # Configure mocks
        mock_frames = [{"frame_number": i, "timestamp": i, "image_data": "base64data", 
                        "scene_change": False, "metadata": {}} for i in range(5)]
        mock_extract_frames.return_value = mock_frames
        
        mock_batches = [{"frames": [i], "image_data": ["base64data"], 
                         "timestamps": [i], "ocr_text": [""]} for i in range(5)]
        mock_create_batches.return_value = mock_batches
        
        mock_analysis_results = [{"batch_id": i, "frames": [i], "timestamps": [i], 
                                 "analysis_text": f"Analysis {i}", "context": f"Context {i}", 
                                 "metadata": {}} for i in range(5)]
        mock_analyze_batch.side_effect = mock_analysis_results
        
        mock_summary_result = {
            "summary_text": "Summary of the video",
            "summary_path": os.path.join(temp_output_dir, "summary.txt"),
            "full_analysis_path": os.path.join(temp_output_dir, "full_analysis.txt"),
            "flowchart_path": None
        }
        mock_generate_summary.return_value = mock_summary_result
        
        # Update config to use the temp directory
        test_config["output_dir"] = temp_output_dir
        
        # Create and run the pipeline
        pipeline = StandardPipeline(test_config)
        pipeline.vision_model = mock_models["vision"]
        pipeline.summary_model = mock_models["summary"]
        
        result = pipeline.run("test_video.mp4")
        
        # Verify the pipeline executed all steps
        mock_extract_frames.assert_called_once()
        mock_create_batches.assert_called_once()
        assert mock_analyze_batch.call_count == 5
        mock_generate_summary.assert_called_once()
        
        # Verify the result
        assert result["summary_text"] == "Summary of the video"
        assert "output_dir" in result
        assert result["output_dir"] == temp_output_dir

    def test_pipeline_error_handling(self, test_config, mock_models):
        """Test error handling in the pipeline."""
        # Create pipeline
        pipeline = StandardPipeline(test_config)
        pipeline.vision_model = mock_models["vision"]
        pipeline.summary_model = mock_models["summary"]
        
        # Test with non-existent video
        with pytest.raises(PipelineError):
            pipeline.run("nonexistent_video.mp4")


class TestSegmentedPipeline:
    """Integration tests for the segmented pipeline."""

    @patch('smolavision.video.extractor.extract_frames')
    @patch('smolavision.batch.creator.create_batches')
    @patch('smolavision.analysis.vision.analyze_batch')
    @patch('smolavision.analysis.summarization.generate_summary')
    def test_pipeline_execution(self, mock_generate_summary, mock_analyze_batch, 
                               mock_create_batches, mock_extract_frames, 
                               test_config, mock_models, temp_output_dir):
        """Test the complete execution of the segmented pipeline."""
        # Configure mocks
        mock_frames = [{"frame_number": i, "timestamp": i, "image_data": "base64data", 
                        "scene_change": False, "metadata": {}} for i in range(5)]
        mock_extract_frames.return_value = mock_frames
        
        mock_batches = [{"frames": [i], "image_data": ["base64data"], 
                         "timestamps": [i], "ocr_text": [""]} for i in range(5)]
        mock_create_batches.return_value = mock_batches
        
        mock_analysis_results = [{"batch_id": i, "frames": [i], "timestamps": [i], 
                                 "analysis_text": f"Analysis {i}", "context": f"Context {i}", 
                                 "metadata": {}} for i in range(5)]
        mock_analyze_batch.side_effect = mock_analysis_results
        
        mock_summary_result = {
            "summary_text": "Summary of the video",
            "summary_path": os.path.join(temp_output_dir, "summary.txt"),
            "full_analysis_path": os.path.join(temp_output_dir, "full_analysis.txt"),
            "flowchart_path": None
        }
        mock_generate_summary.return_value = mock_summary_result
        
        # Update config to use the temp directory
        test_config["output_dir"] = temp_output_dir
        test_config["segment_length"] = 60  # 1 minute segments
        
        # Create and run the pipeline
        pipeline = SegmentedPipeline(test_config)
        pipeline.vision_model = mock_models["vision"]
        pipeline.summary_model = mock_models["summary"]
        
        # Mock video duration
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_video = mock_video_capture.return_value
            mock_video.isOpened.return_value = True
            mock_video.get.side_effect = lambda prop: {
                0: 30.0,  # FPS
                7: 300,   # Frame count (10 seconds)
            }.get(prop, 0)
            
            result = pipeline.run("test_video.mp4")
        
        # Verify the result
        assert result["summary_text"] == "Summary of the video"
        assert "output_dir" in result
        assert result["output_dir"] == temp_output_dir
