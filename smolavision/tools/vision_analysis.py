# smolavision/tools/vision_analysis.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.models.factory import ModelFactory
from smolavision.batch.types import Batch
from smolavision.analysis.vision import analyze_batch, create_analysis_prompt
from smolavision.analysis.types import AnalysisRequest
from smolavision.exceptions import ToolError, ModelError, AnalysisError

logger = logging.getLogger(__name__)

class VisionAnalysisTool(Tool):
    """Tool for analyzing batches of images using a vision model."""

    name: str = "vision_analysis"
    description: str = "Analyze a batch of image data with a text prompt, provide a list of Batches"
    input_type: str = "list[Batch]"
    output_type: str = "str"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VisionAnalysisTool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = ModelFactory.create_vision_model(config.get("model", {}))
        logger.info(f"VisionAnalysisTool initialized with model: {type(self.model).__name__}")

    def use(self, batches: List[Dict[str, Any]], prompt: str = None, previous_context: str = "") -> str:
        """
        Analyze a batch of images with a text prompt.
        
        Args:
            batches: List of batch dictionaries
            prompt: Optional custom prompt for analysis
            previous_context: Context from previous analysis
            
        Returns:
            Analysis result text
            
        Raises:
            ToolError: If vision analysis fails
        """
        try:
            # Convert dictionaries to Batch objects
            typed_batches = [Batch(**batch) for batch in batches]
            
            if not typed_batches:
                logger.warning("No batches provided for analysis")
                return "No batches to analyze"
            
            # For simplicity, we'll analyze the first batch only
            # In a real implementation, you might want to analyze all batches
            batch = typed_batches[0]
            
            logger.info(f"Analyzing batch with {len(batch.image_data)} images")
            
            # Get analysis parameters from config
            language = self.config.get("video", {}).get("language", "English")
            mission = self.config.get("analysis", {}).get("mission", "general")
            
            # Create analysis request
            request = AnalysisRequest(
                batch_id=0,
                frames=batch.frames,
                timestamps=batch.timestamps,
                image_data=batch.image_data,
                ocr_text=batch.ocr_text if hasattr(batch, "ocr_text") else [],
                previous_context=previous_context,
                language=language,
                mission=mission
            )
            
            # Create prompt if not provided
            if not prompt:
                prompt = create_analysis_prompt(request)
            
            # Analyze images
            result = self.model.analyze_images(
                images=batch.image_data,
                prompt=prompt,
                max_tokens=4096
            )
            
            logger.info(f"Analysis completed, result length: {len(result)}")
            return result
            
        except ModelError as e:
            logger.error(f"Model error during vision analysis: {e}")
            raise ToolError(f"Vision analysis failed: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during vision analysis")
            raise ToolError(f"Vision analysis failed: {e}") from e
