# smolavision/ocr/types.py
from typing import Dict, Any
from pydantic import BaseModel, Field

class OCRData(BaseModel):
    """Represents OCR results for a single image."""
    text: str = Field(..., description="Extracted text from the image.")
    confidence: float = Field(..., description="Confidence level of the OCR extraction (0-100).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional OCR metadata.")