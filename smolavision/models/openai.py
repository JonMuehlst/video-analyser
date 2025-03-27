# smolavision/models/openai.py
import logging
import base64
from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

class OpenAIModel(ModelInterface):
    """Implementation for OpenAI models."""

    def __init__(self, api_key: str, model_id: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096):
        """
        Initialize OpenAI model.

        Args:
            api_key: OpenAI API key
            model_id: OpenAI model ID (default: gpt-4o)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for text generation
        """
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized OpenAI model: {self.model_id}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using OpenAI."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs # Pass other kwargs to the OpenAI API
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            raise ModelError(f"OpenAI text generation failed: {e}") from e

    def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """Analyze images with a text prompt using OpenAI's vision models."""
        try:
            content: List[Dict[str, str]] = []
            for image_data in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",  # Encode as data URL (JPEG)
                        "detail": "high" #or low or auto
                    }
                })
            content.append({"type": "text", "text": prompt})

            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": content}]
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                **kwargs # Pass other kwargs to the OpenAI API, like temperature or top_p
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            raise ModelError(f"OpenAI image analysis failed: {e}") from e