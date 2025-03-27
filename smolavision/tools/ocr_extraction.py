# smolavision/tools/ocr_extraction.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.video.types import Frame
from smolavision.ocr.extractor import extract_text
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class OCRExtractionTool(Tool):
    """Tool for extracting text from frames using OCR."""

    name: str = "ocr_extraction"
    description: str = "Extract text from a list of video frames using OCR. Requires a list of frames"
    input_type: str = "list[Frame]"
    output_type: str = "list[Frame]"

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OCRExtractionTool."""
        self.config = config.get("video", {})

    def use(self, frames: List[Dict[str, Any]]) -> str:
        """Extract text from frames using OCR."""
        try:
            # We are receiving a list of dicts, let's use a comprehension for conversion
            typed_frames = [Frame(**frame) for frame in frames]
            extracted_frames = extract_text(typed_frames, language=self.config.get("language", "English"))
            # serialize results
            return str([frame.model_dump() for frame in extracted_frames])
        except Exception as e:
            raise ToolError(f"OCR extraction failed: {e}") from e
