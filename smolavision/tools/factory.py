# smolavision/tools/factory.py
import logging
from typing import Dict, Any, Type, Optional
from smolavision.tools.base import Tool
from smolavision.tools.frame_extraction import FrameExtractionTool
from smolavision.tools.ocr_extraction import OCRExtractionTool
from smolavision.tools.batch_creation import BatchCreationTool
from smolavision.tools.vision_analysis import VisionAnalysisTool
from smolavision.tools.summarization import SummarizationTool
from smolavision.exceptions import ConfigurationError, ToolError

logger = logging.getLogger(__name__)

class ToolFactory:
    """Factory class for creating tool instances."""
    
    # Tool registry mapping tool names to their classes
    _tool_registry: Dict[str, Type[Tool]] = {
        "frame_extraction": FrameExtractionTool,
        "ocr_extraction": OCRExtractionTool,
        "batch_creation": BatchCreationTool,
        "vision_analysis": VisionAnalysisTool,
        "summarization": SummarizationTool
    }

    @classmethod
    def register_tool(cls, tool_name: str, tool_class: Type[Tool]) -> None:
        """
        Register a new tool class.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class to register
        """
        cls._tool_registry[tool_name] = tool_class
        logger.debug(f"Registered tool: {tool_name}")

    @classmethod
    def create_tool(cls, config: Dict[str, Any], tool_name: str) -> Tool:
        """
        Create a tool instance based on the tool name and configuration.

        Args:
            config: Configuration dictionary
            tool_name: Name of the tool to create

        Returns:
            A tool instance

        Raises:
            ConfigurationError: If the tool is not supported
            ToolError: If tool creation fails
        """
        if tool_name not in cls._tool_registry:
            logger.error(f"Unsupported tool: {tool_name}")
            raise ConfigurationError(f"Unsupported tool: {tool_name}")
        
        try:
            tool_class = cls._tool_registry[tool_name]
            logger.info(f"Creating tool: {tool_name}")
            return tool_class(config)
        except Exception as e:
            logger.exception(f"Failed to create tool: {tool_name}")
            raise ToolError(f"Failed to create tool {tool_name}: {e}") from e

    @classmethod
    def list_available_tools(cls) -> Dict[str, str]:
        """
        List all available tools with their descriptions.
        
        Returns:
            Dictionary mapping tool names to their descriptions
        """
        tools = {}
        for tool_name, tool_class in cls._tool_registry.items():
            # Create a temporary instance to get the description
            # This is not ideal but works for simple tools
            try:
                description = tool_class.description
                if isinstance(description, str):
                    tools[tool_name] = description
                else:
                    tools[tool_name] = "No description available"
            except:
                tools[tool_name] = "No description available"
        
        return tools
