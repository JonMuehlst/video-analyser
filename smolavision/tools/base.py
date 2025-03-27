"""
Base tool class for SmolaVision.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get tool description."""
        pass
    
    @property
    @abstractmethod
    def input_type(self) -> str:
        """Get input type description."""
        pass
    
    @property
    @abstractmethod
    def output_type(self) -> str:
        """Get output type description."""
        pass
    
    @abstractmethod
    def use(self, *args, **kwargs) -> Any:
        """
        Execute the tool.
        
        Args:
            *args: Tool-specific positional arguments
            **kwargs: Tool-specific keyword arguments
            
        Returns:
            Tool-specific output
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data against the expected type.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Basic validation - override for more complex validation
        # This is a placeholder for more sophisticated validation
        return True
