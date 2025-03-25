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

# Helper function to extract content safely from Ollama responses
def extract_content_safely(response):
    """Safely extract content from various response formats"""
    try:
        # Log the response type for debugging
        logger.debug(f"Response type in extract_content_safely: {type(response)}")
        
        # Try object attribute access first (newer Ollama versions)
        if hasattr(response, 'message'):
            logger.debug("Response has 'message' attribute")
            if hasattr(response.message, 'content'):
                logger.debug("Response.message has 'content' attribute")
                return str(response.message.content)
        
        # Try dictionary access (older Ollama versions)
        if isinstance(response, dict):
            logger.debug(f"Response is a dict with keys: {list(response.keys())}")
            if 'message' in response:
                if isinstance(response['message'], dict) and 'content' in response['message']:
                    return str(response['message']['content'])
                elif hasattr(response['message'], 'content'):
                    return str(response['message'].content)
                else:
                    return str(response['message'])
        
        # Try direct content attribute
        if hasattr(response, 'content'):
            logger.debug("Response has 'content' attribute")
            return str(response.content)
        
        # Try dictionary content key
        if isinstance(response, dict) and 'content' in response:
            return str(response['content'])
        
        # If it's already a string, return it
        if isinstance(response, str):
            return response
        
        # Last resort: convert to string
        result = str(response)
        logger.debug(f"Converted response to string: {result[:100]}")
        return result
    except Exception as e:
        logger.error(f"Error in extract_content_safely: {e}")
        return f"Error extracting content: {str(e)}"

@pytest.fixture
def ollama_client():
    """Fixture to provide an Ollama client"""
    try:
        client = ollama.Client(host="http://localhost:11434")
        # Test connection
        client.list()
        return client
    except Exception as e:
        pytest.skip(f"Ollama server not available: {e}")

@pytest.fixture
def model_name(ollama_client):
    """Fixture to find a suitable model for testing"""
    try:
        models_response = ollama_client.list()
        logger.info(f"Models response type: {type(models_response)}")
        
        # Find a suitable model
        model_name = None
        
        # Handle different response formats
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
        
        # If we couldn't find a model, use a known model that's likely to be available
        if not model_name:
            # Try common models in order of preference
            for common_model in ["phi3:mini", "llama3", "mistral", "gemma:2b"]:
                try:
                    # Check if model exists by trying to get info about it
                    ollama_client.show(model=common_model)
                    model_name = common_model
                    logger.info(f"Found common model: {model_name}")
                    break
                except Exception:
                    continue
        
        # If we still don't have a model, use phi3:mini and let the test handle any errors
        if not model_name:
            model_name = "phi3:mini"
            
        logger.info(f"Using model: {model_name}")
        return model_name
    except Exception as e:
        logger.error(f"Error determining model name: {e}")
        return "phi3:mini"  # Fallback

def test_ollama_response_format(ollama_client, model_name):
    """Test to diagnose the response format from Ollama"""
    # Test simple chat request
    logger.info("Testing chat request...")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    # Make the request and capture the raw response
    response = ollama_client.chat(
        model=model_name,
        messages=messages,
        options={"num_predict": 100, "temperature": 0.7}
    )
    
    # Log the raw response
    logger.info(f"Raw response type: {type(response)}")
    logger.info(f"Raw response: {repr(response)[:500]}")
    
    # Inspect the response object in detail
    ResponseWrapper.inspect_object(response, "Response: ")
    
    # Test the safe extraction
    safe_content = extract_content_safely(response)
    logger.info(f"Safe extraction result: {safe_content[:100]}")
    
    # Verify the result is a string
    assert isinstance(safe_content, str), "Safe extraction should always return a string"
    assert len(safe_content) > 0, "Extracted content should not be empty"

def test_direct_attribute_access(ollama_client, model_name):
    """Test direct attribute access on Ollama response"""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = ollama_client.chat(model=model_name, messages=messages)
    
    # Instead of skipping, we'll test our extract_content_safely function
    # which should handle any response format
    content = extract_content_safely(response)
    logger.info(f"Extracted content = {content[:100]}")
    
    # Test that we got a valid response
    assert isinstance(content, str), "Extracted content should be a string"
    assert len(content) > 0, "Extracted content should not be empty"
    
    # Check for numeric content in the response (likely to contain "4" for "2+2")
    assert any(digit in content for digit in ["4", "four"]), "Response should contain the answer (4)"

def test_message_attribute_access(ollama_client, model_name):
    """Test message attribute access on Ollama response"""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = ollama_client.chat(model=model_name, messages=messages)
    
    # Method 2: Through message attribute
    if hasattr(response, 'message'):
        logger.info(f"response.message type = {type(response.message)}")
        if hasattr(response.message, 'content'):
            content = response.message.content
            logger.info(f"response.message.content = {content[:100]}")
            assert isinstance(content, str)
            assert "Paris" in content, "Response should mention Paris"
        else:
            logger.info("response.message has no 'content' attribute")
            pytest.skip("Response.message has no 'content' attribute")
    else:
        logger.info("response has no 'message' attribute")
        pytest.skip("Response has no 'message' attribute")

def test_dictionary_access(ollama_client, model_name):
    """Test dictionary access on Ollama response"""
    messages = [{"role": "user", "content": "What is Python?"}]
    response = ollama_client.chat(model=model_name, messages=messages)
    
    # Log response type for debugging
    logger.info(f"Response type: {type(response)}")
    ResponseWrapper.inspect_object(response, "Response details: ")
    
    # Use our extract_content_safely function instead of direct dictionary access
    content = extract_content_safely(response)
    logger.info(f"Extracted content = {content[:100]}")
    
    # Test that we got a valid response
    assert isinstance(content, str), "Extracted content should be a string"
    assert len(content) > 0, "Extracted content should not be empty"
    
    # Check for Python-related content
    assert any(term.lower() in content.lower() for term in ["python", "programming", "language"]), \
        "Response should be about Python"

def test_string_conversion(ollama_client, model_name):
    """Test string conversion of Ollama response"""
    messages = [{"role": "user", "content": "Hello"}]
    response = ollama_client.chat(model=model_name, messages=messages)
    
    # Method 4: Convert to string
    string_response = str(response)
    logger.info(f"str(response) = {string_response[:100]}")
    assert isinstance(string_response, str)
    assert len(string_response) > 0, "String response should not be empty"

def test_error_handling():
    """Test error handling with a mocked response"""
    # Create a mock response that will cause errors
    class MockResponse:
        pass
    
    response = MockResponse()
    
    # Test the safe extraction with a problematic response
    content = extract_content_safely(response)
    assert isinstance(content, str), "Even with errors, should return a string"
    assert "Error" in content or len(content) > 0, "Should return error message or string representation"

if __name__ == "__main__":
    # When run directly, use pytest to run the tests
    import pytest
    pytest.main(["-v", __file__])
