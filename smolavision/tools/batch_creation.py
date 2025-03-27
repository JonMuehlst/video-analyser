# smolavision/tools/batch_creation.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.video.types import Frame
from smolavision.batch.creator import create_batches
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class BatchCreationTool(Tool):
    """Tool for creating batches of frames."""

    name: str = "batch_creation"
    description: str = "Create batches of frames for analysis, provide a list of frames to batch"
    input_type: str = "list[Frame]"
    output_type: str = "list[Batch]"

    def __init__(self, config: Dict[str, Any]):
        """Initialize the BatchCreationTool."""
        self.config = config.get("analysis", {})

    def use(self, frames: List[Dict[str, Any]]) -> str:
        """Create batches from the input frames."""
        try:
            typed_frames = [Frame(**frame) for frame in frames]
            batches = create_batches(
                frames=typed_frames,
                max_batch_size_mb=self.config.get("max_batch_size_mb", 10.0),
                max_images_per_batch=self.config.get("max_images_per_batch", 15),
                overlap_frames=self.config.get("batch_overlap_frames", 2)
            )
            return str([batch.model_dump() for batch in batches])  # convert each Batch to a dictionary
        except Exception as e:
            raise ToolError(f"Batch creation failed: {e}") from e
