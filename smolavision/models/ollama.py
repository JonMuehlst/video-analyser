# smolavision/models/ollama.py
import logging
import json
import httpx
from typing import List, Dict, Any

from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

class OllamaModel(ModelInterface):
    """Implementation for Ollama models."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3", vision_model_name: str = "llava", temperature: float = 0.7, max_tokens: int = 4096, request_timeout: int = 60):
        """
        Initialize Ollama model.

        Args:
            base_url: URL of the Ollama server (default: http://localhost:11434)
            model_name: Name of the text model (default: llama3)
            vision_model_name: Name of the vision model (default: llava)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for text generation
            request_timeout: The request timeout in seconds
        """
        self.base_url = base_url
        self.model_name = model_name
        self.vision_model = vision_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout) # async client, timeout of 60 seconds
        logger.info(f"Initialized Ollama model: {self.model_name} (vision: {self.vision_model})")

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using Ollama."""
        try:
            data = {
                "prompt": prompt,
                "model": self.model_name,
                "stream": False, # we're requesting single output, so disable streaming
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs  # pass other params here like temperature
            }

            async with self.client as client:
                response = await client.post("/api/generate", json=data)

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json().get("response", "")

        except httpx.TimeoutException as e:
            raise ModelError(f"Ollama text generation timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ModelError(f"Ollama text generation failed with status {e.response.status_code}: {e}") from e
        except Exception as e:
            raise ModelError(f"Ollama text generation failed: {e}") from e

    async def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """Analyze images with a text prompt using Ollama."""
        try:
            # Create the message with the image data
            messages = []
            for image_data in images:
                messages.append({
                    "type": "image",
                    "data": image_data # Image data as base64 encoded string
                })

            messages.append({
                "type": "text",
                "content": prompt
            })

            data = {
                "prompt": "", # Prompt is set as messages
                "model": self.vision_model,
                "stream": False, # Request one value
                "format": "json",
                "messages": messages,
                "max_tokens": max_tokens, # Override tokens
                "temperature": self.temperature,
                **kwargs
            }

            async with self.client as client:
                response = await client.post("/api/chat", json=data)

            response.raise_for_status()
            json_response = response.json()
            #Get string of all responses, assuming it's the chat-style API endpoint
            return "".join([message.get("content","") for message in json_response.get("choices",[]) if message.get("content")])


        except httpx.TimeoutException as e:
            raise ModelError(f"Ollama image analysis timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ModelError(f"Ollama image analysis failed with status {e.response.status_code}: {e}") from e
        except Exception as e:
            raise ModelError(f"Ollama image analysis failed: {e}") from e