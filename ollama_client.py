"""
Ollama client for local model inference
"""
import base64
import json
import logging
import time
import requests
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("SmolaVision")

class OllamaClient:
    """Client for interacting with Ollama API for local model inference"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the Ollama client with the base URL"""
        self.base_url = base_url
        logger.info(f"Initialized Ollama client with base URL: {base_url}")
        self._check_connection()
        
    def _check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            logger.info(f"Successfully connected to Ollama server. Available models: {', '.join(models[:5])}" + 
                       (f" and {len(models)-5} more" if len(models) > 5 else ""))
            return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama server at {self.base_url}")
            logger.warning("Make sure Ollama is installed and running (https://ollama.com/)")
            return False
        except Exception as e:
            logger.warning(f"Error checking Ollama connection: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            result = response.json()
            models = [model["name"] for model in result.get("models", [])]
            logger.info(f"Available Ollama models: {', '.join(models)}")
            return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []
    
    def _make_request(self, url: str, payload: Dict[str, Any], 
                     retries: int = 3, retry_delay: float = 2.0) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Make a request to Ollama API with retries"""
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=120)  # Longer timeout for vision models
                response.raise_for_status()
                return True, response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout calling Ollama API (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling Ollama API: {str(e)} (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error calling Ollama API: {str(e)}")
                return False, f"Error: {str(e)}"
        
        return False, "Failed after multiple retries"
        
    def generate(self, 
                 model: str, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 max_tokens: int = 4096) -> str:
        """Generate text using an Ollama model"""
        url = f"{self.base_url}/api/generate"
        
        # Adjust parameters based on model size to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if "phi" in model or "gemma:2b" in model:
            adjusted_max_tokens = min(max_tokens, 2048)  # Smaller limit for smaller models
        elif "tiny" in model or "1b" in model:
            adjusted_max_tokens = min(max_tokens, 1024)  # Even smaller limit for tiny models
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": adjusted_max_tokens
            }
        }
        
        # Add GPU-specific options to prevent OOM errors
        if "3060" in model or any(small_model in model for small_model in ["phi", "gemma:2b", "tiny", "1b", "7b"]):
            payload["options"].update({
                "gpu_layers": -1,  # Use all layers that fit on GPU
                "f16": True,       # Use half-precision for better memory usage
            })
        
        if system_prompt:
            payload["system"] = system_prompt
            
        logger.debug(f"Sending request to Ollama: {model} (max tokens: {adjusted_max_tokens})")
        success, result = self._make_request(url, payload)
        
        if success:
            return result.get("response", "")
        else:
            return f"Error generating text with Ollama: {result}"
    
    def generate_vision(self, 
                        model: str, 
                        prompt: str, 
                        images: List[str],
                        max_tokens: int = 4096) -> str:
        """Generate text using an Ollama vision model with images
        
        Args:
            model: The Ollama vision model name (e.g., "llava")
            prompt: The text prompt
            images: List of base64-encoded images
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # First check if the model supports chat API (newer Ollama versions)
        try:
            # Try using the chat API first (preferred for newer Ollama versions)
            return self._generate_vision_chat(model, prompt, images, max_tokens)
        except Exception as e:
            logger.warning(f"Chat API failed for vision model, falling back to generate API: {str(e)}")
            # Fall back to the generate API
            return self._generate_vision_legacy(model, prompt, images, max_tokens)
    
    def _generate_vision_chat(self, model: str, prompt: str, images: List[str], max_tokens: int = 4096) -> str:
        """Generate vision response using the chat API (newer Ollama versions)"""
        url = f"{self.base_url}/api/chat"
        
        # Adjust parameters based on model and number of images to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if len(images) > 2:
            # Reduce token limit for multiple images
            adjusted_max_tokens = min(max_tokens, 2048)
        
        # Format images for Ollama chat API
        content = [{"type": "text", "text": prompt}]
        
        for img_base64 in images:
            content.append({
                "type": "image",
                "image": {
                    "data": img_base64,
                    "mime_type": "image/jpeg"
                }
            })
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": False,
            "options": {
                "num_predict": adjusted_max_tokens,
                "gpu_layers": -1,  # Use all layers that fit on GPU
                "f16": True        # Use half-precision for better memory usage
            }
        }
        
        logger.debug(f"Sending vision chat request to Ollama: {model} with {len(images)} images (max tokens: {adjusted_max_tokens})")
        success, result = self._make_request(url, payload)
        
        if success:
            return result.get("message", {}).get("content", "")
        else:
            return f"Error generating vision response with Ollama chat API: {result}"
    
    def _generate_vision_legacy(self, model: str, prompt: str, images: List[str], max_tokens: int = 4096) -> str:
        """Generate vision response using the legacy generate API"""
        url = f"{self.base_url}/api/generate"
        
        # Adjust parameters based on model and number of images to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if len(images) > 2:
            # Reduce token limit for multiple images
            adjusted_max_tokens = min(max_tokens, 2048)
        
        # Format images for Ollama generate API
        formatted_images = []
        for img_base64 in images:
            formatted_images.append({
                "data": img_base64,
                "mime_type": "image/jpeg"
            })
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": formatted_images,
            "options": {
                "num_predict": adjusted_max_tokens,
                "gpu_layers": -1,  # Use all layers that fit on GPU
                "f16": True        # Use half-precision for better memory usage
            }
        }
        
        logger.debug(f"Sending vision generate request to Ollama: {model} with {len(images)} images (max tokens: {adjusted_max_tokens})")
        success, result = self._make_request(url, payload)
        
        if success:
            return result.get("response", "")
        else:
            return f"Error generating vision response with Ollama generate API: {result}"
