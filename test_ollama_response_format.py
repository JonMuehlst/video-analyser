#!/usr/bin/env python
"""
Test file to diagnose the 'str' object has no attribute 'content' error
with Ollama responses
"""

import os
import sys
import logging
import pytest
import ollama
import json
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level to see all response details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OllamaResponseTest")

class ResponseWrapper:
    """Wrapper to help debug response objects"""
    
    @staticmethod
    def inspect_object(obj: Any, prefix: str = "") -> None:
        """Inspect an object and log its attributes and methods"""
        logger.debug(f"{prefix}Type: {type(obj)}")
        
        # If it's a dictionary, log its keys and values
        if isinstance(obj, dict):
            logger.debug(f"{prefix}Dict keys: {list(obj.keys())}")
            for key, value in obj.items():
                logger.debug(f"{prefix}  {key}: {type(value)} = {repr(value)[:100]}")
            return
            
        # If it's a list, log its items
        if isinstance(obj, list):
            logger.debug(f"{prefix}List with {len(obj)} items")
            for i, item in enumerate(obj[:5]):  # Show first 5 items
                logger.debug(f"{prefix}  [{i}]: {type(item)} = {repr(item)[:100]}")
            return
            
        # For other objects, log attributes and methods
        attrs = dir(obj)
        logger.debug(f"{prefix}Attributes: {[a for a in attrs if not a.startswith('__')]}")
        
        # Try to access common attributes safely
        for attr in ['content', 'message', 'text', 'response']:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                logger.debug(f"{prefix}obj.{attr} = {type(value)} {repr(value)[:100]}")
                
                # Recursively inspect nested objects
                if not isinstance(value, (str, int, float, bool, type(None))):
                    ResponseWrapper.inspect_object(value, prefix + "  ")

def test_ollama_response_format():
    """Test to diagnose the response format from Ollama"""
    try:
        # Initialize Ollama client
        client = ollama.Client(host="http://localhost:11434")
        logger.info("Connected to Ollama server")
        
        # Get available models
        models_response = client.list()
        logger.info(f"Models response type: {type(models_response)}")
        
        # Find a suitable model for testing
        model_name = None
        if isinstance(models_response, dict) and 'models' in models_response:
            models = models_response['models']
            if models and isinstance(models[0], dict):
                if 'name' in models[0]:
                    model_name = models[0]['name']
                elif 'model' in models[0]:
                    model_name = models[0]['model']
        elif isinstance(models_response, list) and models_response:
            if isinstance(models_response[0], dict):
                if 'name' in models_response[0]:
                    model_name = models_response[0]['name']
                elif 'model' in models_response[0]:
                    model_name = models_response[0]['model']
        
        if not model_name:
            model_name = "phi3:mini"  # Fallback to a common model
            
        logger.info(f"Using model: {model_name}")
        
        # Test simple chat request
        logger.info("Testing chat request...")
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        # Make the request and capture the raw response
        response = client.chat(
            model=model_name,
            messages=messages,
            options={"num_predict": 100, "temperature": 0.7}
        )
        
        # Log the raw response
        logger.info(f"Raw response type: {type(response)}")
        logger.info(f"Raw response: {repr(response)[:500]}")
        
        # Inspect the response object in detail
        ResponseWrapper.inspect_object(response, "Response: ")
        
        # Test different ways to access the content
        logger.info("Testing different ways to access content...")
        
        # Method 1: Direct attribute access
        try:
            if hasattr(response, 'content'):
                logger.info(f"response.content = {response.content[:100]}")
            else:
                logger.info("response has no 'content' attribute")
        except Exception as e:
            logger.error(f"Error accessing response.content: {e}")
            
        # Method 2: Through message attribute
        try:
            if hasattr(response, 'message'):
                logger.info(f"response.message type = {type(response.message)}")
                if hasattr(response.message, 'content'):
                    logger.info(f"response.message.content = {response.message.content[:100]}")
                else:
                    logger.info("response.message has no 'content' attribute")
            else:
                logger.info("response has no 'message' attribute")
        except Exception as e:
            logger.error(f"Error accessing response.message.content: {e}")
            
        # Method 3: Dictionary access
        try:
            if isinstance(response, dict):
                if 'message' in response:
                    logger.info(f"response['message'] type = {type(response['message'])}")
                    if isinstance(response['message'], dict) and 'content' in response['message']:
                        logger.info(f"response['message']['content'] = {response['message']['content'][:100]}")
                    else:
                        logger.info("response['message'] has no 'content' key or is not a dict")
                else:
                    logger.info("response dict has no 'message' key")
            else:
                logger.info("response is not a dict")
        except Exception as e:
            logger.error(f"Error accessing response['message']['content']: {e}")
            
        # Method 4: Convert to string
        try:
            string_response = str(response)
            logger.info(f"str(response) = {string_response[:100]}")
        except Exception as e:
            logger.error(f"Error converting response to string: {e}")
            
        # Test the safe extraction function
        def extract_content_safely(response):
            """Safely extract content from various response formats"""
            try:
                # Try object attribute access first (newer Ollama versions)
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    return response.message.content
                
                # Try dictionary access (older Ollama versions)
                if isinstance(response, dict):
                    if 'message' in response:
                        if isinstance(response['message'], dict) and 'content' in response['message']:
                            return response['message']['content']
                        elif hasattr(response['message'], 'content'):
                            return response['message'].content
                        else:
                            return str(response['message'])
                
                # Try direct content attribute
                if hasattr(response, 'content'):
                    return response.content
                
                # Try dictionary content key
                if isinstance(response, dict) and 'content' in response:
                    return response['content']
                
                # If it's already a string, return it
                if isinstance(response, str):
                    return response
                
                # Last resort: convert to string
                return str(response)
            except Exception as e:
                logger.error(f"Error in extract_content_safely: {e}")
                return f"Error extracting content: {str(e)}"
        
        # Test the safe extraction
        safe_content = extract_content_safely(response)
        logger.info(f"Safe extraction result: {safe_content[:100]}")
        
        # Verify the result is a string
        assert isinstance(safe_content, str), "Safe extraction should always return a string"
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_ollama_response_format()
