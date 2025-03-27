# smolavision/tools/vision_analysis.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.models.factory import create_vision_model
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class VisionAnalysisTool(Tool):
    """Tool for analyzing batches of images using a vision model."""

    name: str = "vision_analysis"
    description: str = "Analyze a batch of image data with a text prompt, provide a list of Batches"
    input_type: str = "list[Batch]"
    output_type: str = "str"  # The model should be returning JSON

    def __init__(self, config: Dict[str, Any]):
        """Initialize the VisionAnalysisTool."""
        self.config = config
        self.model = create_vision_model(config.get("model", {}))

    def use(self, batches: List[Dict[str, Any]], prompt: str, previous_context: str = "") -> str:
        """Analyze a batch of images with a text prompt."""
        try:
            # It needs to convert back a list of Dictionaries to a list of Batches
            from smolavision.batch.types import Batch
            typed_batches = [Batch(**batch) for batch in batches]

            # Extract all of the images and run analysis
            image_data = []
            for batch in typed_batches:
                image_data.extend(batch.image_data)

            result = self.model.analyze_images(images=image_data, prompt=prompt, max_tokens=4096)
            return result  # Assuming that the Model.analyze images result provides useful context

        except Exception as e:
            raise ToolError(f"Vision analysis failed: {e}") from e
