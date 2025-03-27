# smolavision/video/types.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Frame(BaseModel):
    """Represents a single frame extracted from a video."""
    frame_number: int = Field(..., description="Sequential frame number in the video.")
    timestamp: float = Field(..., description="Timestamp of the frame in seconds.")
    image_data: str = Field(..., description="Base64 encoded image data of the frame.")  # Store as base64
    scene_change: bool = Field(False, description="Indicates if this frame is a scene change.")
    ocr_text: Optional[str] = Field(None, description="Extracted OCR text from the frame, if OCR is enabled.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the frame.")