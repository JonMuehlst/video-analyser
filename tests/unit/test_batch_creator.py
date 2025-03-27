import pytest
from unittest.mock import patch

from smolavision.batch.creator import create_batches
from smolavision.video.types import Frame
from smolavision.batch.types import Batch


class TestBatchCreator:
    """Tests for the batch creation functionality."""

    def test_create_batches_empty_frames(self):
        """Test that create_batches handles empty frame list."""
        result = create_batches([])
        assert result == []

    def test_create_batches_single_batch(self, sample_frames):
        """Test creating a single batch when all frames fit."""
        # Set a large max batch size so all frames fit in one batch
        batches = create_batches(
            frames=sample_frames,
            max_batch_size_mb=100.0,
            max_images_per_batch=10
        )
        
        # Verify results
        assert len(batches) == 1
        assert len(batches[0].frames) == len(sample_frames)
        assert batches[0].frames == [frame.frame_number for frame in sample_frames]

    def test_create_batches_multiple_by_count(self, sample_frames):
        """Test creating multiple batches due to max_images_per_batch limit."""
        # Set max_images_per_batch to 1 to force multiple batches
        batches = create_batches(
            frames=sample_frames,
            max_batch_size_mb=100.0,
            max_images_per_batch=1,
            overlap_frames=0
        )
        
        # Verify results
        assert len(batches) == len(sample_frames)
        assert batches[0].frames == [sample_frames[0].frame_number]
        assert batches[1].frames == [sample_frames[1].frame_number]
        assert batches[2].frames == [sample_frames[2].frame_number]

    def test_create_batches_with_overlap(self, sample_frames):
        """Test creating batches with overlap frames."""
        # Set max_images_per_batch to 1 and overlap to 1
        batches = create_batches(
            frames=sample_frames,
            max_batch_size_mb=100.0,
            max_images_per_batch=1,
            overlap_frames=1
        )
        
        # Verify results
        assert len(batches) == len(sample_frames)
        assert batches[0].frames == [sample_frames[0].frame_number]
        # Second batch should include first frame as overlap
        assert batches[1].frames == [sample_frames[0].frame_number, sample_frames[1].frame_number]
        # Third batch should include second frame as overlap
        assert batches[2].frames == [sample_frames[1].frame_number, sample_frames[2].frame_number]

    @patch('smolavision.batch.utils.calculate_batch_size_mb')
    def test_create_batches_by_size(self, mock_calculate_size, sample_frames):
        """Test creating batches based on size limit."""
        # Mock the size calculation to return a large value for the second frame
        mock_calculate_size.side_effect = lambda batch: 5.0 if len(batch) > 1 else 1.0
        
        # Set max_batch_size_mb to 4.0 to force a split after the first frame
        batches = create_batches(
            frames=sample_frames,
            max_batch_size_mb=4.0,
            max_images_per_batch=10,
            overlap_frames=0
        )
        
        # Verify results
        assert len(batches) == 2
        assert batches[0].frames == [sample_frames[0].frame_number]
        assert batches[1].frames == [sample_frames[1].frame_number, sample_frames[2].frame_number]

    def test_create_batches_with_ocr_text(self, sample_frames):
        """Test creating batches with OCR text."""
        # Add OCR text to frames
        for i, frame in enumerate(sample_frames):
            frame.ocr_text = f"OCR text {i}"
        
        batches = create_batches(
            frames=sample_frames,
            max_batch_size_mb=100.0,
            max_images_per_batch=10
        )
        
        # Verify results
        assert len(batches) == 1
        assert batches[0].ocr_text == ["OCR text 0", "OCR text 1", "OCR text 2"]
