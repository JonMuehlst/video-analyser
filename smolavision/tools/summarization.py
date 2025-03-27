# smolavision/tools/summarization.py
import logging
import os
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.models.factory import ModelFactory
from smolavision.analysis.summarization import create_summary_prompt, generate_summary
from smolavision.exceptions import ToolError, ModelError, SummarizationError

logger = logging.getLogger(__name__)

class SummarizationTool(Tool):
    """Tool for generating a coherent summary from batch analyses."""

    name: str = "summarization"
    description: str = "Generate a summary of the analysis results, provide a list of analysis texts"
    input_type: str = "list[str]"
    output_type: str = "str"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SummarizationTool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = ModelFactory.create_summary_model(config.get("model", {}))
        logger.info(f"SummarizationTool initialized with model: {type(self.model).__name__}")

    def use(self, analyses: List[str], language: str = None, mission: str = None, generate_flowchart: bool = None) -> str:
        """
        Generate a summary of the analysis results.
        
        Args:
            analyses: List of analysis texts
            language: Language for the summary (defaults to config value)
            mission: Analysis mission type (defaults to config value)
            generate_flowchart: Whether to generate a flowchart (defaults to config value)
            
        Returns:
            Generated summary text
            
        Raises:
            ToolError: If summarization fails
        """
        try:
            # Get parameters from config if not provided
            if language is None:
                language = self.config.get("video", {}).get("language", "English")
            
            if mission is None:
                mission = self.config.get("analysis", {}).get("mission", "general")
            
            if generate_flowchart is None:
                generate_flowchart = self.config.get("analysis", {}).get("generate_flowchart", False)
            
            logger.info(f"Generating summary for {len(analyses)} analyses "
                       f"(language: {language}, mission: {mission}, flowchart: {generate_flowchart})")
            
            # Create output directory if it doesn't exist
            output_dir = self.config.get("output_dir", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate summary
            result = generate_summary(
                analyses=analyses,
                language=language,
                mission=mission,
                generate_flowchart=generate_flowchart,
                model=self.model,
                output_dir=output_dir
            )
            
            logger.info(f"Summary generated, length: {len(result.get('summary_text', ''))}")
            return result.get("summary_text", "")
            
        except ModelError as e:
            logger.error(f"Model error during summarization: {e}")
            raise ToolError(f"Summarization failed: {e}") from e
        except SummarizationError as e:
            logger.error(f"Summarization error: {e}")
            raise ToolError(f"Summarization failed: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during summarization")
            raise ToolError(f"Summarization failed: {e}") from e
