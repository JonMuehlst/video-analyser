# smolavision/tools/ocr_extraction.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.video.types import Frame
from smolavision.ocr.extractor import extract_text
from smolavision.exceptions import ToolError, OCRProcessingError

logger = logging.getLogger(__name__)

class OCRExtractionTool(Tool):
    """Tool for extracting text from frames using OCR."""

    name: str = "ocr_extraction"
    description: str = "Extract text from a list of video frames using OCR. Requires a list of frames"
    input_type: str = "List[Dict[str, Any]]" # Input is now list of dicts
    output_type: str = "List[Dict[str, Any]]" # Output is now list of dicts

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OCRExtractionTool."""
        self.config = config.get("video", {})

    def use(self, frames: List[Dict[str, Any]]) -> str:
        """
        Extract text from frames using OCR.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            String representation of frames with extracted OCR text
            
        Raises:
            ToolError: If OCR extraction fails
        """
        try:
            # Convert dictionaries to Frame objects
            typed_frames = [Frame(**frame) for frame in frames]
            
            # Extract text using OCR
            language = self.config.get("language", "English")
            logger.info(f"Extracting OCR text from {len(typed_frames)} frames using language: {language}")
            
            extracted_frames = extract_text(typed_frames, language=language)
            
            # Return the actual list of frame dictionaries
            return [frame.model_dump() for frame in extracted_frames]
            
        except OCRProcessingError as e:
            logger.error(f"OCR processing error: {e}")
            raise ToolError(f"OCR extraction failed: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during OCR extraction")
            raise ToolError(f"OCR extraction failed: {e}") from e
