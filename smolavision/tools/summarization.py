"""
Summarization tool adapter for SmolaVision.
"""

import logging
from typing import List, Dict, Any, Optional

from ..exceptions import SummarizationError
from ..logging import get_logger
from ..analysis.summary import generate_summary
from ..models.base import ModelInterface

logger = get_logger("tools.summarization")


class SummarizationTool:
    """Adapter for summarization functionality"""
    
    @staticmethod
    def execute(
        analyses: List[str],
        model: ModelInterface,
        language: str = "English",
        mission: str = "general",
        generate_flowchart: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a coherent summary from all batch analyses.
        
        Args:
            analyses: List of all batch analyses
            model: Model to use for summarization
            language: Language of text in the video
            mission: Specific analysis mission (e.g., 'workflow', 'general')
            generate_flowchart: Whether to generate a flowchart diagram
            output_dir: Directory to save output files
            
        Returns:
            Dictionary containing summary results
            
        Raises:
            SummarizationError: If summarization fails
        """
        try:
            return generate_summary(
                analyses=analyses,
                model=model,
                language=language,
                mission=mission,
                generate_flowchart=generate_flowchart,
                output_dir=output_dir
            )
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg)
            raise SummarizationError(error_msg) from e
