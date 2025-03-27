# smolavision/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ModelInterface(ABC):
    """Base interface for all AI models."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """Analyze images with a text prompt.

        Args:
            images: List of base64-encoded images
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Generated analysis text
        """
        pass