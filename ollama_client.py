"""
Client for interacting with Ollama API for local model inference
"""
import os
import json
import base64
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
            logger.info(f"Available Ollama models: {', '.join(models) if models else 'None'}")
            return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []
    
    def _make_request(self, url: str, payload: Dict[str, Any], 
                     retries: int = 3, retry_delay: float = 2.0) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Make a request to Ollama API with retries"""
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=300)  # Longer timeout for vision models
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
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            response = requests.get(f"{self.base_url}/api/show?name={model_name}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting model info: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            logger.info(f"Pulling model: {model_name}...")
            
            # Use the Ollama API to pull the model
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}
            
            # This is a long operation, so we'll stream the response
            response = requests.post(url, json=payload, stream=True, timeout=3600)
            
            if response.status_code != 200:
                logger.error(f"Error pulling model: {response.status_code} - {response.text}")
                return False
            
            # Process the streaming response to show progress
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            if data.get("completed", False):
                                logger.info(f"Model pull completed: {model_name}")
                                return True
                            else:
                                # Show progress if available
                                progress = data.get("progress", 0)
                                total = data.get("total", 0)
                                if total > 0:
                                    percent = (progress / total) * 100
                                    logger.info(f"Pulling {model_name}: {percent:.1f}% ({progress}/{total})")
                    except json.JSONDecodeError:
                        pass
            
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
        
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
        if any(small_model in model.lower() for small_model in ["phi", "gemma:2b", "tiny", "1b", "7b"]):
            payload["options"].update({
                "gpu_layers": -1,  # Use all layers that fit on GPU
                "f16": True,       # Use half-precision for better memory usage
            })
        
        if system_prompt:
            payload["system"] = system_prompt
            
        logger.info(f"Generating text with model: {model} (max tokens: {adjusted_max_tokens})")
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
        
        # Adjust parameters based on model and number of images to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if len(images) > 2:
            # Reduce token limit for multiple images
            adjusted_max_tokens = min(max_tokens, 2048)
        
        # Format images for Ollama
        formatted_images = []
        for img_base64 in images:
            # Remove data:image/jpeg;base64, prefix if present
            if "base64," in img_base64:
                img_base64 = img_base64.split("base64,")[1]
                
            formatted_images.append({
                "data": img_base64,
                "type": "image/jpeg"
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
        
        logger.info(f"Generating vision response with model: {model} with {len(images)} images")
        success, result = self._make_request(url, payload)
        
        if success:
            return result.get("response", "")
        else:
            return f"Error generating vision response with Ollama: {result}"
