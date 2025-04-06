# smolavision/models/ollama.py
import logging
import json
import httpx
import time
from typing import List, Dict, Any, Optional

from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

class OllamaModel(ModelInterface):
    """Implementation for Ollama models for local inference."""

    def __init__(self, 
                 base_url: str = "http://localhost:11434", 
                 model_name: str = "llama3", 
                 vision_model_name: str = "llava", 
                 temperature: float = 0.7, 
                 max_tokens: int = 4096, 
                 request_timeout: int = 60,
                 max_retries: int = 3,
                 retry_delay: int = 2):
        """
        Initialize Ollama model.

        Args:
            base_url: URL of the Ollama server (default: http://localhost:11434)
            model_name: Name of the text model (default: llama3)
            vision_model_name: Name of the vision model (default: llava)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for text generation
            request_timeout: The request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url
        self.model_name = model_name
        self.vision_model = vision_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize HTTP clients
        self.client = httpx.Client(base_url=self.base_url, timeout=self.request_timeout)
        self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout)
        
        # Check connection to Ollama server
        self._check_connection()
        
        logger.info(f"Initialized Ollama model: {self.model_name} (vision: {self.vision_model})")

    def _check_connection(self) -> bool:
        """
        Check connection to Ollama server.
        
        Returns:
            True if connection is successful, raises exception otherwise
        
        Raises:
            ModelError: If connection to Ollama server fails
        """
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise ModelError(f"Failed to connect to Ollama server at {self.base_url}: {e}") from e

    def _make_request_with_retry(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Ollama API with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response JSON
            
        Raises:
            ModelError: If request fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(endpoint, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.warning(f"Request timed out (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ModelError(f"Request timed out after {self.max_retries} attempts: {e}") from e
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ModelError(f"HTTP error {e.response.status_code} after {self.max_retries} attempts: {e}") from e
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ModelError(f"Unexpected error after {self.max_retries} attempts: {e}") from e
            
            # Wait before retrying
            time.sleep(self.retry_delay)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using Ollama.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated text
            
        Raises:
            ModelError: If text generation fails
        """
        try:
            logger.debug(f"Generating text with model {self.model_name}, prompt length: {len(prompt)}")
            
            data = {
                "prompt": prompt,
                "model": self.model_name,
                "stream": False,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = value
            
            # Make request with retry logic
            response_json = self._make_request_with_retry("/api/generate", data)
            
            # Extract response text
            response_text = response_json.get("response", "")
            logger.debug(f"Text generation completed, response length: {len(response_text)}")
            
            return response_text

        except ModelError:
            # Re-raise ModelError exceptions
            raise
        except Exception as e:
            logger.exception("Unexpected error during text generation")
            raise ModelError(f"Ollama text generation failed: {e}") from e

    def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """
        Analyze images with a text prompt using Ollama.
        
        Args:
            images: List of base64-encoded images
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated analysis text
            
        Raises:
            ModelError: If image analysis fails
        """
        try:
            logger.debug(f"Analyzing {len(images)} images with model {self.vision_model}")
            
            # Create messages with image data
            messages = []
            
            # Limit the number of images to prevent overloading the model
            max_images = kwargs.get("max_images", 10)
            if len(images) > max_images:
                logger.warning(f"Too many images ({len(images)}), limiting to {max_images}")
                images = images[:max_images]
            
            # Add image messages
            for image_data in images:
                messages.append({
                    "type": "image",
                    "data": image_data
                })
            
            # Add prompt message
            messages.append({
                "type": "text",
                "content": prompt
            })
            
            data = {
                "model": self.vision_model,
                "stream": False,
                "format": "json",
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in data and key not in ["max_images"]:
                    data[key] = value
            
            # Make request with retry logic
            response_json = self._make_request_with_retry("/api/chat", data)
            
            # Extract response content from choices
            choices = response_json.get("choices", [])
            if not choices:
                logger.warning("No choices in response")
                return ""
            
            # Concatenate all content from choices
            response_text = "".join([
                message.get("content", "") 
                for message in choices 
                if message.get("content")
            ])
            
            logger.debug(f"Image analysis completed, response length: {len(response_text)}")
            return response_text

        except ModelError:
            # Re-raise ModelError exceptions
            raise
        except Exception as e:
            logger.exception("Unexpected error during image analysis")
            raise ModelError(f"Ollama image analysis failed: {e}") from e

    # Async methods for compatibility with async code
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using Ollama (asynchronous).
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated text
            
        Raises:
            ModelError: If text generation fails
        """
        # For now, we'll use the synchronous version
        # TODO: Implement true async request using self.async_client
        logger.warning("Using synchronous call within async method generate_text_async")
        return self.generate_text(prompt, **kwargs)

    async def analyze_images_async(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """
        Analyze images with a text prompt using Ollama (asynchronous).
        
        Args:
            images: List of base64-encoded images
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated analysis text
            
        Raises:
            ModelError: If image analysis fails
        """
        # TODO: Implement true async request using self.async_client
        logger.warning("Using synchronous call within async method analyze_images_async")
        return self.analyze_images(images, prompt, max_tokens, **kwargs)
