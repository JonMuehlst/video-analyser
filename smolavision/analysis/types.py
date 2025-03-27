from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    """Request for batch analysis."""
    batch_id: int = Field(..., description="Unique identifier for the batch.")
    frames: List[int] = Field(..., description="List of frame numbers in the batch.")
    timestamps: List[float] = Field(..., description="List of timestamps for the frames.")
    image_data: List[str] = Field(..., description="List of base64 encoded image data.")
    ocr_text: List[str] = Field(default_factory=list, description="List of OCR text for each frame.")
    previous_context: str = Field("", description="Context from previous analysis.")
    language: str = Field("English", description="Language for analysis output.")
    mission: str = Field("general", description="Analysis mission type (general or workflow).")

class AnalysisResult(BaseModel):
    """Result of batch analysis."""
    batch_id: int = Field(..., description="Unique identifier for the batch.")
    frames: List[int] = Field(..., description="List of frame numbers in the batch.")
    timestamps: List[float] = Field(..., description="List of timestamps for the frames.")
    analysis_text: str = Field(..., description="Generated analysis text.")
    context: str = Field(..., description="Context for next batch analysis.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
