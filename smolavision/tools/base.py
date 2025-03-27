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
    def input_schema(self) -> Dict[str, Any]:
        """Get input schema."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Get output schema."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool-specific input parameters
            
        Returns:
            Tool-specific output
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters against the schema.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Basic validation - override for more complex validation
        for key, schema in self.input_schema.items():
            if schema.get("required", False) and key not in kwargs:
                raise ValueError(f"Required parameter '{key}' missing")
        return True
