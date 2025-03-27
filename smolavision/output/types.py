from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class OutputFormat(str, Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"

class OutputResult(BaseModel):
    """Result of output generation."""
    summary: str = Field(..., description="Summary text.")
    full_analysis: str = Field(..., description="Full analysis text.")
    flowchart: Optional[str] = Field(None, description="Flowchart in Mermaid syntax.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    files: Dict[str, str] = Field(default_factory=dict, description="Mapping of file types to file paths.")
