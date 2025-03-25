#!/usr/bin/env python
"""
Test file for OllamaModel integration with smolagents
"""

import os
import sys
import logging
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OllamaModelTest")

class TestOllamaModel:
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

def test_string_input():
    """Test with a simple string input"""
    model = TestOllamaModel()
    result = model("Hello, how are you?")
    print(f"String input result: {result}")
    assert isinstance(result, str), "Result should be a string"
    return True

def test_dict_input():
    """Test with a dictionary input"""
    model = TestOllamaModel()
    result = model({"role": "user", "content": "What is the capital of France?"})
    print(f"Dict input result: {result}")
    assert isinstance(result, str), "Result should be a string"
    return True

def test_list_input():
    """Test with a list of messages"""
    model = TestOllamaModel()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ]
    result = model(messages)
    print(f"List input result: {result}")
    assert isinstance(result, str), "Result should be a string"
    return True

def main():
    """Run all tests"""
    print("Testing OllamaModel integration with smolagents")
    
    try:
        # Check if Ollama server is running
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        print(f"Connected to Ollama server, available models: {[m['name'] for m in models.get('models', [])]}")
    except Exception as e:
        print(f"Error connecting to Ollama server: {e}")
        print("Please make sure Ollama is installed and running")
        return False
    
    # Run tests
    tests = [
        ("String input test", test_string_input),
        ("Dict input test", test_dict_input),
        ("List input test", test_list_input)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} ERROR: {str(e)}")
            all_passed = False
    
    if all_passed:
        print("\nAll tests PASSED! The OllamaModel integration should work correctly.")
    else:
        print("\nSome tests FAILED. Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
