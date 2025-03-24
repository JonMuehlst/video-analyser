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
            logger.info("Successfully connected to Ollama server")
            return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama server at {self.base_url}")
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
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        logger.debug(f"Sending request to Ollama: {model}")
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
        url = f"{self.base_url}/api/generate"
        
        # Format images for Ollama
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
                "num_predict": max_tokens
            }
        }
        
        logger.debug(f"Sending vision request to Ollama: {model} with {len(images)} images")
        success, result = self._make_request(url, payload)
        
        if success:
            return result.get("response", "")
        else:
            return f"Error generating vision response with Ollama: {result}"
