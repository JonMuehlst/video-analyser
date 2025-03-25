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
            logger.info(f"Available Ollama models: {', '.join(models) if models else 'None'}")
            return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []

    def _extract_content_safely(self, response: Any, default_message: str = "No content") -> str:
        """Safely extract content from various Ollama response formats

        This helper handles all known response formats from different Ollama versions
        and ensures a string is always returned.

        Args:
            response: The response object from Ollama
            default_message: Message to return if no content can be extracted

        Returns:
            Extracted content as a string
        """
        try:
            # Log response type for debugging
            logger.debug(f"Extracting content from response type: {type(response)}")

            # If it's already a string, return it
            if isinstance(response, str):
                return response

            # Handle dictionary response
            if isinstance(response, dict):
                # Try message.content path
                if 'message' in response:
                    message = response['message']
                    if isinstance(message, dict) and 'content' in message:
                        return str(message['content'])
                    return str(message)  # Just return message as string

                # Try direct content
                if 'content' in response:
                    return str(response['content'])

                # Try response field (common in generate API)
                if 'response' in response:
                    return str(response['response'])

            # Handle object with attributes
            if hasattr(response, 'message'):
                message = response.message
                if hasattr(message, 'content'):
                    return str(message.content)
                return str(message)

            # Try direct content attribute
            if hasattr(response, 'content'):
                return str(response.content)

            # Try response attribute
            if hasattr(response, 'response'):
                return str(response.response)

            # Last resort: convert entire response to string
            result = str(response)
            if result and result != '{}':
                return result

            return default_message

        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return f"Error extracting content: {str(e)}"

    def _make_request(self, url: str, payload: Dict[str, Any], 
                     retries: int = 3, retry_delay: float = 2.0) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Make a request to Ollama API with retries"""
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=120)  # Longer timeout for vision models

                # Check for error responses even with 200 status
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Check for error field in response
                        if isinstance(result, dict) and "error" in result:
                            error_msg = result["error"]
                            logger.error(f"Ollama API returned error: {error_msg}")

                            # If it's a validation error, return it specially so we can handle it
                            if "validation" in error_msg.lower() or "pydantic" in error_msg.lower():
                                return False, f"Validation error: {error_msg}"

                            if attempt < retries - 1:
                                time.sleep(retry_delay)
                                continue
                            return False, error_msg
                        return True, result
                    except ValueError:
                        # Not JSON, return the text
                        return True, response.text

                response.raise_for_status()
                return True, response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout calling Ollama API (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling Ollama API: {str(e)} (attempt {attempt+1}/{retries})")
                # Check if the error response contains JSON
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict) and "error" in error_data:
                            error_msg = error_data["error"]
                            # If it's a validation error, return it specially
                            if "validation" in error_msg.lower() or "pydantic" in error_msg.lower():
                                return False, f"Validation error: {error_msg}"
                            return False, error_msg
                    except:
                        pass

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
        if "3060" in model or any(small_model in model.lower() for small_model in ["phi", "gemma:2b", "tiny", "1b", "7b"]):
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
            Generated text response as a string
        """
        url = f"{self.base_url}/api/generate"

        # Adjust parameters based on model and number of images to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if len(images) > 2:
            # Reduce token limit for multiple images
            adjusted_max_tokens = min(max_tokens, 2048)

        # Format images for Ollama generate API
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
                "f16": True  # Use half-precision for better memory usage
            }
        }

        logger.info(f"Generating vision response with model: {model} with {len(images)} images")
        success, result = self._make_request(url, payload)

        if success:
            # IMPORTANT: Always ensure we return a string
            if isinstance(result, str):
                return result

            # If it's a dict with a response key (typical for generate API)
            if isinstance(result, dict) and "response" in result:
                return str(result["response"])

            # Extract content from any response format using our helper
            if hasattr(self, '_extract_content_safely'):
                return self._extract_content_safely(result)

            # Fallback if _extract_content_safely doesn't exist
            try:
                # Try different paths to content
                if isinstance(result, dict):
                    if 'message' in result:
                        if isinstance(result['message'], dict) and 'content' in result['message']:
                            return str(result['message']['content'])
                        return str(result['message'])
                    return str(result)

                if hasattr(result, 'message'):
                    if hasattr(result.message, 'content'):
                        return str(result.message.content)
                    return str(result.message)

                if hasattr(result, 'content'):
                    return str(result.content)

                # Last resort: convert to string
                return str(result)
            except Exception as e:
                logger.error(f"Error extracting content: {str(e)}")
                return f"Error: {str(e)}"
        else:
            return f"Error generating vision response with Ollama: {result}"

    def _generate_vision_chat(self, model: str, prompt: str, images: List[str], max_tokens: int = 4096) -> str:
        """Generate vision response using the chat API (newer Ollama versions)"""
        url = f"{self.base_url}/api/chat"

        # Adjust parameters based on model and number of images to prevent OOM errors
        adjusted_max_tokens = max_tokens
        if len(images) > 2:
            # Reduce token limit for multiple images
            adjusted_max_tokens = min(max_tokens, 2048)

        # Format images for Ollama chat API
        # Some Ollama versions expect a string for content, not a list
        try:
            # First try with the standard format (list of content items)
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
                if isinstance(result, dict) and "message" in result:
                    if isinstance(result["message"], dict) and "content" in result["message"]:
                        return result["message"]["content"]
                    elif hasattr(result["message"], "content"):
                        return result["message"].content
                    else:
                        return str(result["message"])
                return str(result)
            else:
                # If we get a validation error, try the fallback approach
                if "validation error" in str(result).lower() or "pydantic" in str(result).lower():
                    logger.warning("Validation error with content list format, trying string format")
                    return self._generate_vision_chat_fallback(model, prompt, images, adjusted_max_tokens)
                return f"Error generating vision response with Ollama chat API: {result}"
        except Exception as e:
            logger.warning(f"Error with standard vision format: {str(e)}, trying fallback")
            return self._generate_vision_chat_fallback(model, prompt, images, adjusted_max_tokens)

    def _generate_vision_chat_fallback(self, model: str, prompt: str, images: List[str], max_tokens: int = 4096) -> str:
        """Fallback method for vision generation using string content format"""
        url = f"{self.base_url}/api/chat"

        # For the fallback, we'll use a text-only approach with image descriptions
        image_descriptions = [f"[Image {i+1}]" for i in range(len(images))]
        combined_prompt = f"{prompt}\n\nThe message includes {len(images)} images: {', '.join(image_descriptions)}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": combined_prompt
                }
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "gpu_layers": -1,
                "f16": True
            }
        }

        logger.debug(f"Sending fallback vision chat request to Ollama: {model}")
        success, result = self._make_request(url, payload)

        if success:
            if isinstance(result, dict) and "message" in result:
                if isinstance(result["message"], dict) and "content" in result["message"]:
                    return result["message"]["content"]
                elif hasattr(result["message"], "content"):
                    return result["message"].content
                else:
                    return str(result["message"])
            return str(result)
        else:
            # If even the fallback fails, try the legacy method
            logger.warning("Fallback vision chat failed, trying legacy generate method")
            return self._generate_vision_legacy(model, prompt, images, max_tokens)

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
            return f"Error generating vision response with Ollama generate API: {result}"
