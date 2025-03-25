#!/usr/bin/env python
"""
OCR extraction tool for SmolaVision
"""

import logging
from typing import List, Dict, Any
import re
import base64
from io import BytesIO

from smolagents import Tool

# Configure logging
logger = logging.getLogger("SmolaVision")


class OCRExtractionTool(Tool):
    """Tool for extracting text from frames using OCR"""
    name = "extract_text_ocr"
    description = "Extracts text from frames using OCR with support for Hebrew"
    inputs = {
        "frames": {
            "type": "array",
            "description": "List of extracted frames to process with OCR"
        },
        "language": {
            "type": "string",
            "description": "Primary language to optimize OCR for (e.g., 'heb' for Hebrew)",
            "nullable": True
        }
    }
    output_type = "array"

    def forward(self, frames: List[Dict], language: str = "heb") -> List[Dict]:
        """Extract text from frames using Tesseract OCR"""
        logger.info(f"Extracting text using OCR from {len(frames)} frames")

        try:
            import pytesseract
            from PIL import Image

            # Map common language names to Tesseract language codes
            language_map = {
                "Hebrew": "heb",
                "English": "eng",
                "Arabic": "ara",
                "Russian": "rus",
                # Add more mappings as needed
            }

            # Ensure language is not None before using it
            if language is None:
                language = "eng"
                logger.warning("Language parameter was None, defaulting to English")

            # Get the Tesseract language code
            lang_code = language_map.get(language, language)

            # Process each frame with OCR
            for i, frame in enumerate(frames):
                if "base64_image" not in frame:
                    logger.warning(f"Frame {i} missing base64_image, skipping OCR")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = "Missing base64_image"
                    continue

                # Ensure the base64_image is a string before decoding
                base64_image = frame.get("base64_image", "")
                if not isinstance(base64_image, str) or not base64_image:
                    logger.warning(f"Frame {i} has invalid base64_image data, skipping OCR")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = "Invalid base64 image data"
                    continue

                try:
                    # Decode base64 image
                    img_data = base64.b64decode(base64_image)
                    img = Image.open(BytesIO(img_data))

                    # Run OCR with specified language
                    text = pytesseract.image_to_string(img, lang=lang_code)

                    # Clean up text (remove excessive whitespace)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Add the extracted text to the frame data
                    frame["ocr_text"] = text
                    frame["ocr_language"] = lang_code

                    # Log a preview of the text (first 50 chars)
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    logger.debug(f"Frame {i} OCR: {text_preview}")

                except Exception as e:
                    logger.error(f"OCR failed for frame {i}: {str(e)}")
                    frame["ocr_text"] = ""
                    frame["ocr_error"] = str(e)

            logger.info(f"OCR extraction completed for {len(frames)} frames")
            return frames

        except ImportError:
            logger.error("Required OCR libraries not installed. Install with: pip install pytesseract pillow")
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_error"] = "OCR libraries not installed"
            return frames
        except Exception as e:
            error_msg = f"Error during OCR processing: {str(e)}"
            logger.error(error_msg)
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_error"] = error_msg
            return frames
