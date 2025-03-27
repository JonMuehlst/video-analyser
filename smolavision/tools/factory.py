# smolavision/tools/factory.py
import logging
from typing import Dict, Any
from smolavision.tools.base import Tool
from smolavision.tools.frame_extraction import FrameExtractionTool
from smolavision.tools.ocr_extraction import OCRExtractionTool
from smolavision.tools.batch_creation import BatchCreationTool
from smolavision.tools.vision_analysis import VisionAnalysisTool
from smolavision.tools.summarization import SummarizationTool
from smolavision.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class ToolFactory:
    """Factory class for creating tool instances."""

    @staticmethod
    def create_tool(config: Dict[str, Any], tool_name: str) -> Tool:
        """
        Create a tool instance based on the tool name and configuration.

        Args:
            config: Configuration dictionary.
            tool_name: Name of the tool to create.

        Returns:
            A tool instance.

        Raises:
            ConfigurationError: If the tool is not supported.
        """
        if tool_name == "frame_extraction":
            return FrameExtractionTool(config)
        elif tool_name == "ocr_extraction":
            return OCRExtractionTool(config)
        elif tool_name == "batch_creation":
            return BatchCreationTool(config)
        elif tool_name == "vision_analysis":
            return VisionAnalysisTool(config)
        elif tool_name == "summarization":
            return SummarizationTool(config)
        else:
            raise ConfigurationError(f"Unsupported tool: {tool_name}")