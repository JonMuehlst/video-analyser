# smolavision/models/anthropic.py
import logging
from typing import List, Dict, Any
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

class AnthropicModel(ModelInterface):
    """Implementation for Anthropic Claude models."""

    def __init__(self, api_key: str, model_id: str = "claude-3-opus-20240229", temperature: float = 0.7, max_tokens: int = 4096):
        """
        Initialize Anthropic model.

        Args:
            api_key: Anthropic API key
            model_id: Anthropic model ID (default: claude-3-opus-20240229)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for text generation
        """
        self.client = Anthropic(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized Anthropic model: {self.model_id}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using Anthropic Claude."""
        try:
            completion = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{ "role": "user", "content": prompt }],
                temperature=self.temperature,
                **kwargs  # Pass any other kwargs to the Anthropic API
            )
            if completion.content and len(completion.content) > 0:
                return completion.content[0].text
            else:
                return ""
        except Exception as e:
            raise ModelError(f"Anthropic text generation failed: {e}") from e

    def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """Analyze images with a text prompt using Anthropic Claude."""
        try:
            messages: List[Dict[str, Any]] = []  # Ensure proper typing
            for image_data in images:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Hardcode assuming JPEG
                                "data": image_data
                            }
                        },
                    ],
                })


            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            completion = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=messages,
                temperature=self.temperature,
                **kwargs # Add top_p top_k etc here.
            )

            if completion.content and len(completion.content) > 0:
                return completion.content[0].text
            else:
                return ""
        except Exception as e:
            raise ModelError(f"Anthropic image analysis failed: {e}") from e