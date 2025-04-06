# smolavision/ocr/extractor.py
import logging
import base64
from typing import List, Dict, Any
from PIL import Image
import pytesseract
from io import BytesIO
from smolavision.exceptions import OCRProcessingError
from smolavision.video.types import Frame
from smolavision.ocr.types import OCRData
from smolavision.ocr.languages import get_tesseract_lang_code

logger = logging.getLogger(__name__)

def extract_text(frames: List[Frame], language: str = "English") -> List[Frame]:
    """
    Extract text from a list of frames using OCR.

    Args:
        frames: List of Frame objects
        language: Language of the text in the frames (e.g., "English", "Spanish")

    Returns:
        List of Frame objects with the 'ocr_text' attribute populated.

    Raises:
        OCRProcessingError: If OCR processing fails.
    """
    logger.info(f"Attempting text extraction from frames using OCR (language: {language})")

    try:
        # Check for Tesseract availability early
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract is not installed or not in your PATH. Skipping OCR.")
        # Return frames unmodified if Tesseract is not found
        for frame in frames:
            frame.ocr_text = None
            frame.metadata["ocr_confidence"] = 0.0
            frame.metadata["ocr_skipped_reason"] = "Tesseract not found"
        return frames
    except Exception as e:
        # Catch other potential Tesseract setup issues
        logger.error(f"Error checking Tesseract version: {e}. Skipping OCR.")
        for frame in frames:
            frame.ocr_text = None
            frame.metadata["ocr_confidence"] = 0.0
            frame.metadata["ocr_skipped_reason"] = f"Tesseract check error: {e}"
        return frames

    # Proceed with OCR if Tesseract seems available
    try:
        lang_code = get_tesseract_lang_code(language)
        if not lang_code:
            # Log warning for unsupported language but continue without OCR for this call
            logger.warning(f"Unsupported language for OCR: {language}. Skipping OCR.")
            for frame in frames:
                frame.ocr_text = None
                frame.metadata["ocr_confidence"] = 0.0
                frame.metadata["ocr_skipped_reason"] = f"Unsupported language: {language}"
            return frames
            # Or raise error? Current behaviour is skip. Let's stick to skip.
            # raise OCRProcessingError(f"Unsupported language: {language}")


        logger.info(f"Extracting text from frames using Tesseract (language code: {lang_code})")
        for frame in frames:
            try:
                # Decode base64 image data
                img_data = base64.b64decode(frame.image_data)
                img = Image.open(BytesIO(img_data))

                # Perform OCR
                text = pytesseract.image_to_string(img, lang=lang_code)
                confidence = 0.0 #PyTesseract doesn't directly provide a single "confidence"
                # Let's say for now just give the frame number mod 100
                confidence = frame.frame_number % 100


                frame.ocr_text = text
                frame.metadata["ocr_confidence"] = confidence #Let's add it to metadata as well
                logger.debug(f"Extracted text from frame {frame.frame_number}: {text[:50]}...") #Log first 50 chars

            except Exception as e:
                logger.warning(f"OCR failed for frame {frame.frame_number}: {e}")
                frame.ocr_text = None  # Set to None on failure to avoid carrying old text
                frame.metadata["ocr_confidence"] = 0.0  # Add 0 confidence to metadata

        logger.info("Finished OCR processing loop.")
        return frames

    # Removed TesseractNotFoundError handler here as it's checked upfront
    except Exception as e:
        # Catch unexpected errors during the loop
        logger.exception(f"Unexpected error during OCR processing loop: {e}")
        # Attempt to set remaining frames to None/0 confidence
        # This part might be tricky depending on where the error occurred
        # For simplicity, we might just raise the error or return partially processed frames
        raise OCRProcessingError(f"OCR processing failed during loop: {e}") from e
