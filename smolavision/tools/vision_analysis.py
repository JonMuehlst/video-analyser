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
                logger.warning("No batches provided for analysis")
                return "No batches to analyze"
            
            # Analyze ALL batches provided
            all_analysis_texts = []
            current_context = previous_context # Start with initial context

            for i, batch_dict in enumerate(typed_batches):
                batch = Batch(**batch_dict)
                logger.info(f"Analyzing batch {i+1}/{len(typed_batches)} with {len(batch.image_data)} images")

                # Get analysis parameters from config for this batch
                language = self.config.get("video", {}).get("language", "English")
                mission = self.config.get("analysis", {}).get("mission", "general")

                # Create analysis request for the current batch
                request = AnalysisRequest(
                    batch_id=i, # Use the loop index for batch_id
                    frames=batch.frames,
                    timestamps=batch.timestamps,
                    image_data=batch.image_data,
                    ocr_text=batch.ocr_text if hasattr(batch, "ocr_text") else [],
                    previous_context=current_context, # Use the updated context
                    language=language,
                    mission=mission
                )

                # Create prompt if not provided (use context from previous batch)
                analysis_prompt = prompt if prompt else create_analysis_prompt(request)

                # Analyze images for the current batch
            analysis_text = self.model.analyze_images(
                images=batch.image_data,
                prompt=analysis_prompt,
                max_tokens=4096 # Consider making this configurable
            )

            # Extract context for the *next* batch
            # Use a utility function if available, otherwise simple truncation
            from smolavision.analysis.utils import extract_context
            current_context = extract_context(analysis_text) # Update context for the next iteration

            # Append analysis text for the current batch (correct indentation)
            all_analysis_texts.append(analysis_text)
            logger.debug(f"Batch {i+1} analysis completed, result length: {len(analysis_text)}")

            # Combine results from all batches (simple concatenation for now) - Indented correctly
            final_analysis = "\n\n--- Batch Break ---\n\n".join(all_analysis_texts)
            logger.info(f"Completed analysis of {len(typed_batches)} batches. Total length: {len(final_analysis)}")
            return final_analysis

        except ModelError as e:
            logger.error(f"Model error during vision analysis: {e}")
            raise ToolError(f"Vision analysis failed: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during vision analysis")
            raise ToolError(f"Vision analysis failed: {e}") from e
