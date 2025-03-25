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
        
        # Find a suitable model
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
                    model_name = models[0]['name']
                elif 'model' in models_response[0]:
                    model_name = models[0]['model']
        
        if not model_name:
            model_name = "phi3:mini"  # Fallback to a common model
            
        logger.info(f"Using model: {model_name}")
        return model_name
    except Exception as e:
        pytest.skip(f"Could not determine model name: {e}")
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
    
    # Method 1: Direct attribute access
    if hasattr(response, 'content'):
        content = response.content
        logger.info(f"response.content = {content[:100]}")
        assert isinstance(content, str)
    else:
        logger.info("response has no 'content' attribute")
        pytest.skip("Response has no 'content' attribute")

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
    
    # Method 3: Dictionary access
    if isinstance(response, dict):
        if 'message' in response:
            logger.info(f"response['message'] type = {type(response['message'])}")
            if isinstance(response['message'], dict) and 'content' in response['message']:
                content = response['message']['content']
                logger.info(f"response['message']['content'] = {content[:100]}")
                assert isinstance(content, str)
                assert any(term in content.lower() for term in ["python", "programming", "language"]), \
                    "Response should be about Python"
            else:
                logger.info("response['message'] has no 'content' key or is not a dict")
                pytest.skip("Response['message'] has no 'content' key or is not a dict")
        else:
            logger.info("response dict has no 'message' key")
            pytest.skip("Response dict has no 'message' key")
    else:
        logger.info("response is not a dict")
        pytest.skip("Response is not a dict")

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
