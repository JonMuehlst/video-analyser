# smolavision/ocr/__init__.py
from smolavision.ocr.extractor import extract_text
from smolavision.ocr.types import OCRData
from smolavision.ocr.languages import get_tesseract_lang_code

__all__ = [
    "extract_text",
    "OCRData",
    "get_tesseract_lang_code",
]