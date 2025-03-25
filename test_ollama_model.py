#!/usr/bin/env python
"""
Test file for OllamaModel integration with smolagents
"""

import os
import sys
import logging
import pytest
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OllamaModelTest")

class OllamaModel:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        logger.info(f"Initialized TestOllamaModel with model: {model_name}")
    
    def __call__(self, messages, **kwargs):
        # This matches the interface expected by smolagents
        try:
            # Format messages properly for Ollama
            formatted_messages = []
            if isinstance(messages, str):
                formatted_messages.append({"role": "user", "content": messages})
            elif isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, str):
                        formatted_messages.append({"role": "user", "content": msg})
                    elif isinstance(msg, dict) and 'content' in msg:
                        formatted_messages.append(msg)
                    else:
                        logger.warning(f"Skipping invalid message format: {msg}")
            else:
                formatted_messages.append({"role": "user", "content": str(messages)})
            
            # Extract relevant parameters
            max_tokens = kwargs.get("max_tokens", 4096)
            temperature = kwargs.get("temperature", 0.7)
            
            # Call Ollama API directly
            response = self.client.chat(
                model=self.model_name,
                messages=formatted_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            
            # Handle response properly
            if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                logger.error(f"Unexpected response format from Ollama: {response}")
                return f"Error: Unexpected response format from Ollama"
        except Exception as e:
            logger.error(f"Error in OllamaModel.__call__: {str(e)}")
            return f"Error calling Ollama API: {str(e)}"

@pytest.fixture
def check_ollama_server():
    """Fixture to check if Ollama server is running"""
    try:
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        
        # Debug the structure of the response
        logger.info(f"Ollama models response structure: {models}")
        
        # Handle different response formats
        if isinstance(models, dict) and 'models' in models:
            # New format
            if isinstance(models['models'], list):
                if models['models'] and isinstance(models['models'][0], dict):
                    # Try to extract model names based on available keys
                    if 'name' in models['models'][0]:
                        available_models = [m['name'] for m in models['models']]
                    elif 'model' in models['models'][0]:
                        available_models = [m['model'] for m in models['models']]
                    else:
                        # Just use the first key as identifier
                        first_key = next(iter(models['models'][0]))
                        available_models = [m.get(first_key, str(m)) for m in models['models']]
                else:
                    available_models = [str(m) for m in models['models']]
            else:
                available_models = ["<models format not recognized>"]
        elif isinstance(models, list):
            # Direct list format
            if models and isinstance(models[0], dict):
                if 'name' in models[0]:
                    available_models = [m['name'] for m in models]
                elif 'model' in models[0]:
                    available_models = [m['model'] for m in models]
                else:
                    # Just use the first key as identifier
                    first_key = next(iter(models[0]))
                    available_models = [m.get(first_key, str(m)) for m in models]
            else:
                available_models = [str(m) for m in models]
        else:
            available_models = ["<unknown format>"]
            
        logger.info(f"Connected to Ollama server, available models: {available_models}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Ollama server: {e}")
        pytest.skip("Ollama server not available. Please make sure Ollama is installed and running.")

@pytest.fixture
def ollama_model():
    """Fixture to create an OllamaModel instance"""
    return OllamaModel()

def test_string_input(check_ollama_server, ollama_model):
    """Test with a simple string input"""
    result = ollama_model("Hello, how are you?")
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Response should not be empty"

def test_dict_input(check_ollama_server, ollama_model):
    """Test with a dictionary input"""
    result = ollama_model({"role": "user", "content": "What is the capital of France?"})
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Response should not be empty"
    assert "Paris" in result.lower(), "Response should mention Paris as the capital of France"

def test_list_input(check_ollama_server, ollama_model):
    """Test with a list of messages"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
    result = ollama_model(messages)
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Response should not be empty"
    assert any(term in result.lower() for term in ["python", "programming", "language"]), \
        "Response should be about Python programming"

def test_error_handling(ollama_model, monkeypatch):
    """Test error handling when API call fails"""
    # Patch the client.chat method to raise an exception
    def mock_chat(*args, **kwargs):
        raise Exception("Simulated API error")
    
    monkeypatch.setattr(ollama_model.client, "chat", mock_chat)
    
    result = ollama_model("This should trigger an error")
    assert "Error" in result, "Response should contain error information"

def test_message_formatting():
    """Test message formatting without requiring Ollama server"""
    model = OllamaModel()
    
    # Mock the client.chat method to just return the formatted messages
    def mock_chat(model, messages, options=None):
        return {"message": {"content": f"Received: {messages}"}}
    
    # Save original method
    original_chat = model.client.chat
    
    try:
        # Replace with mock
        model.client.chat = mock_chat
        
        # Test string input
        result = model("test message")
        assert "Received:" in result
        assert "test message" in result
        
        # Test dict input
        result = model({"role": "user", "content": "dict message"})
        assert "dict message" in result
        
        # Test list input
        result = model([
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user message"}
        ])
        assert "system prompt" in result
        assert "user message" in result
        
    finally:
        # Restore original method
        model.client.chat = original_chat
