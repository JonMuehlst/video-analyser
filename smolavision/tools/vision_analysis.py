"""
Vision analysis tool adapter for SmolaVision.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from ..exceptions import ModelAPIError
from ..logging import get_logger
from ..analysis.vision import analyze_batch
from ..models.base import ModelInterface

logger = get_logger("tools.vision_analysis")


class VisionAnalysisTool:
    """Adapter for vision analysis functionality"""
    
    @staticmethod
    def execute(
        batch: List[Dict[str, Any]],
        model: ModelInterface,
        previous_context: str = "",
        language: str = "English",
        mission: str = "general",
        output_dir: Optional[str] = None
    ) -> str:
        """
        Analyze a batch of frames using a vision model.
        
        Args:
            batch: Batch of frames to analyze
            model: Vision model to use for analysis
            previous_context: Context from previous analysis for continuity
            language: Language of text in the video
            mission: Specific analysis mission (e.g., 'workflow', 'general')
            output_dir: Directory to save output files
            
        Returns:
            Analysis result as text
            
        Raises:
            ModelAPIError: If the model API call fails
        """
        try:
            return analyze_batch(
                batch=batch,
                model=model,
                previous_context=previous_context,
                language=language,
                mission=mission,
                output_dir=output_dir
            )
        except Exception as e:
            error_msg = f"Error analyzing batch: {str(e)}"
            logger.error(error_msg)
            raise ModelAPIError(error_msg) from e
