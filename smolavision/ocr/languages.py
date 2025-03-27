# smolavision/ocr/languages.py
import logging

logger = logging.getLogger(__name__)

# A mapping of common language names to Tesseract language codes.
# This is not exhaustive, but covers common cases.
TESSERACT_LANGUAGE_MAPPING = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Chinese": "chi_sim",  # Simplified Chinese
    "Japanese": "jpn",
    "Russian": "rus",
    "Arabic": "ara",
    "Hindi": "hin",
}

def get_tesseract_lang_code(language: str) -> str | None:
    """
    Get the Tesseract language code for a given language name.

    Args:
        language: The name of the language (e.g., "English", "Spanish").

    Returns:
        The corresponding Tesseract language code (e.g., "eng", "spa"), or None
        if the language is not supported.
    """
    code = TESSERACT_LANGUAGE_MAPPING.get(language)
    if not code:
        logger.warning(f"Language '{language}' not supported for OCR.")
    return code