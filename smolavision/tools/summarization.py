# smolavision/tools/summarization.py
import logging
from typing import Dict, Any, List
from smolavision.tools.base import Tool
from smolavision.models.factory import create_summary_model
from smolavision.exceptions import ToolError

logger = logging.getLogger(__name__)

class SummarizationTool(Tool):
    """Tool for generating a summary of the analysis results."""

    name: str = "summarization"
    description: str = "Generate a summary of the analysis results, provide a string from vision analysis"
    input_type: str = "str"
    output_type: str = "str"

    def __init__(self, config: Dict[str, Any]):
        """Initialize the SummarizationTool."""
        self.config = config
        self.model = create_summary_model(config.get("model", {}))

    def use(self, analyses: List[str], language: str = "English") -> str:
        """Generate a summary of the analysis results."""
        try:
            # Concatenate all analyses
            full_text = "\n".join(analyses)
            # Create a prompt for summarization
            prompt = f"Summarize the following text in {language}:\n{full_text}"
            # Generate the summary
            summary = self.model.generate_text(prompt)
            return summary
        except Exception as e:
            raise ToolError(f"Summarization failed: {e}") from e
