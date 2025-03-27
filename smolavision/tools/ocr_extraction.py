"""
OCR extraction tool adapter for SmolaVision.
"""

import logging
from typing import List, Dict, Any

from ..exceptions import OCRError
from ..logging import get_logger
from ..ocr.extractor import extract_text

logger = get_logger("tools.ocr_extraction")


class OCRExtractionTool:
    """Adapter for OCR extraction functionality"""
    
    @staticmethod
    def execute(
        frames: List[Dict[str, Any]],
        language: str = "eng"
    ) -> List[Dict[str, Any]]:
        """
        Extract text from frames using Tesseract OCR.
        
        Args:
            frames: List of frames to process
            language: Primary language to optimize OCR for
            
        Returns:
            Updated list of frames with OCR text
            
        Raises:
            OCRError: If OCR processing fails
        """
        try:
            # Map common language names to Tesseract language codes
            language_map = {
                "Hebrew": "heb",
                "English": "eng",
                "Arabic": "ara",
                "Russian": "rus",
                # Add more mappings as needed
            }
            
            # Ensure language is not None
            if language is None:
                language = "eng"
                logger.warning("Language parameter was None, defaulting to English")
                
            # Get the Tesseract language code
            lang_code = language_map.get(language, language)
            
            return extract_text(frames=frames, language=lang_code)
        except Exception as e:
            error_msg = f"Error during OCR processing: {str(e)}"
            logger.error(error_msg)
            raise OCRError(error_msg) from e
