# smolavision/batch/types.py
from typing import List
from pydantic import BaseModel, Field

class Batch(BaseModel):
    """Represents a batch of frames for analysis."""
    frames: List[int] = Field(..., description="List of frame numbers included in the batch.")
    image_data: List[str] = Field(..., description="List of base64 encoded image data for the frames in the batch.")
    timestamps: List[float] = Field(..., description="List of timestamps for the frames in the batch.")
    ocr_text: List[str] = Field(default_factory=tuple, description="List of OCR Text in respective frames.")