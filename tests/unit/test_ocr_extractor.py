import pytest
from unittest.mock import patch, MagicMock
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

from smolavision.ocr.extractor import extract_text
from smolavision.video.types import Frame
from smolavision.exceptions import OCRProcessingError


class TestOCRExtractor:
    """Tests for the OCR extraction functionality."""

    def create_text_image(self, text="Test", size=(200, 100)):
        """Create a test image with text."""
        image = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text, fill='black')
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def test_extract_text_empty_frames(self):
        """Test that extract_text handles empty frame list."""
        result = extract_text([])
        assert result == []

    @patch('pytesseract.image_to_string')
    def test_extract_text_basic(self, mock_image_to_string):
        """Test basic OCR extraction functionality."""
        # Mock pytesseract response
        mock_image_to_string.return_value = "Extracted text"
        
        # Create test frames
        frames = [
            Frame(
                frame_number=0,
                timestamp=0.0,
                image_data=self.create_text_image("Hello"),
                scene_change=False,
                metadata={}
            ),
            Frame(
                frame_number=1,
                timestamp=1.0,
                image_data=self.create_text_image("World"),
                scene_change=False,
                metadata={}
            )
        ]
        
        # Call the function
        result = extract_text(frames, language="English")
        
        # Verify results
        assert len(result) == 2
        assert result[0].ocr_text == "Extracted text"
        assert result[1].ocr_text == "Extracted text"
        assert "ocr_confidence" in result[0].metadata

    @patch('pytesseract.image_to_string')
    def test_extract_text_with_error(self, mock_image_to_string):
        """Test OCR extraction with an error in one frame."""
        # Mock pytesseract to succeed on first call and fail on second
        mock_image_to_string.side_effect = [
            "Extracted text",
            Exception("OCR error")
        ]
        
        # Create test frames
        frames = [
            Frame(
                frame_number=0,
                timestamp=0.0,
                image_data=self.create_text_image("Hello"),
                scene_change=False,
                metadata={}
            ),
            Frame(
                frame_number=1,
                timestamp=1.0,
                image_data=self.create_text_image("World"),
                scene_change=False,
                metadata={}
            )
        ]
        
        # Call the function
        result = extract_text(frames, language="English")
        
        # Verify results
        assert len(result) == 2
        assert result[0].ocr_text == "Extracted text"
        assert result[1].ocr_text is None  # Should be None due to error
        assert result[1].metadata["ocr_confidence"] == 0.0

    @patch('pytesseract.image_to_string')
    def test_extract_text_language_mapping(self, mock_image_to_string):
        """Test OCR extraction with different languages."""
        mock_image_to_string.return_value = "Extracted text"
        
        # Create test frame
        frame = Frame(
            frame_number=0,
            timestamp=0.0,
            image_data=self.create_text_image("Hello"),
            scene_change=False,
            metadata={}
        )
        
        # Call the function with different languages
        extract_text([frame], language="English")
        mock_image_to_string.assert_called_with(pytest.approx(anything), lang="eng")
        
        extract_text([frame], language="Spanish")
        mock_image_to_string.assert_called_with(pytest.approx(anything), lang="spa")

    @patch('pytesseract.image_to_string', side_effect=Exception("Tesseract not found"))
    def test_extract_text_tesseract_not_found(self, mock_image_to_string):
        """Test OCR extraction when Tesseract is not installed."""
        frame = Frame(
            frame_number=0,
            timestamp=0.0,
            image_data=self.create_text_image("Hello"),
            scene_change=False,
            metadata={}
        )
        
        with pytest.raises(OCRProcessingError):
            extract_text([frame], language="English")


# Helper for comparing PIL images in assertions
def anything(obj):
    """Match anything in pytest assertions."""
    return True
