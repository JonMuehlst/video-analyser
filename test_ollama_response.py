import pytest
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("OllamaResponseTest")


class OllamaModel:
    """Minimal version of the OllamaModel class for testing"""

    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.client = None

    def __call__(self, messages, **kwargs):
        """Process messages and return a response from the Ollama model."""
        try:
            # Format messages properly for Ollama
            formatted_messages = []

            # Handle different message formats
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
                # Handle other types of input by converting to string
                formatted_messages.append({"role": "user", "content": str(messages)})

            # Extract relevant parameters
            max_tokens = kwargs.get("max_tokens", 4096)
            temperature = kwargs.get("temperature", 0.7)

            # Log what we're about to send
            logger.debug(f"Calling Ollama API with model: {self.model_name}")
            logger.debug(f"Number of messages: {len(formatted_messages)}")

            # Call Ollama API
            response = self.client.chat(
                model=self.model_name,
                messages=formatted_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )

            # Log raw response type for debugging
            logger.debug(f"Raw Ollama response type: {type(response)}")
            if isinstance(response, str):
                logger.debug(f"String response from Ollama (first 100 chars): {response[:100]}")
            else:
                logger.debug(f"Non-string response from Ollama: {str(response)[:100]}")

            # IMPORTANT: Always check for string first
            if isinstance(response, str):
                return response

            # Handle dictionary response
            if isinstance(response, dict):
                if 'message' in response:
                    message = response['message']
                    if isinstance(message, dict) and 'content' in message:
                        return str(message['content'])
                    # Handle case where message itself might be a string
                    return str(message)
                elif 'content' in response:
                    return str(response['content'])
                elif 'response' in response:
                    return str(response['response'])

            # Handle object with attributes - wrap in try/except to catch AttributeErrors
            try:
                if hasattr(response, 'message'):
                    message = response.message
                    if hasattr(message, 'content'):
                        return str(message.content)
                    # Handle case where message might not have content
                    return str(message)

                if hasattr(response, 'content'):
                    return str(response.content)

                if hasattr(response, 'response'):
                    return str(response.response)
            except AttributeError as attr_err:
                logger.error(f"AttributeError accessing response: {str(attr_err)}")
                return f"Error accessing response attribute: {str(attr_err)}"

            # Last resort: convert to string
            return str(response)

        except Exception as e:
            logger.error(f"Error in OllamaModel.__call__: {str(e)}")
            return f"Error calling Ollama API: {str(e)}"


@pytest.fixture
def model():
    """Fixture that provides an OllamaModel instance"""
    return OllamaModel(model_name="test-model")


def test_string_response_handling(model):
    """Test that the OllamaModel.__call__ method correctly handles string responses"""

    # Mock the client
    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns a string directly
            return "This is a direct string response from Ollama"

    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the string response correctly
    assert response == "This is a direct string response from Ollama"
    assert isinstance(response, str)


def test_object_response_handling(model):
    """Test that the OllamaModel.__call__ method correctly handles object responses"""

    # Define a mock response object with a message attribute that has a content attribute
    class MockMessage:
        def __init__(self):
            self.content = "Response from message.content attribute"

    class MockResponse:
        def __init__(self):
            self.message = MockMessage()

    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns an object with nested attributes
            return MockResponse()

    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the response correctly extracted from the object structure
    assert response == "Response from message.content attribute"
    assert isinstance(response, str)


def test_dict_response_handling(model):
    """Test that the OllamaModel.__call__ method correctly handles dictionary responses"""

    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns a dictionary
            return {
                "message": {
                    "content": "Response from message.content in dictionary"
                }
            }

    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the response correctly extracted from the dictionary
    assert response == "Response from message.content in dictionary"
    assert isinstance(response, str)


def test_error_case_handling(model):
    """Test that the OllamaModel.__call__ method handles errors gracefully"""

    class MockClient:
        def chat(self, model, messages, options=None):
            # Create a response that will trigger an attribute error
            class BrokenResponse:
                @property
                def message(self):
                    raise AttributeError("'BrokenResponse' object has no attribute 'content'")

            return BrokenResponse()

    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got an error message instead of a crash
    assert "Error" in response
    assert isinstance(response, str)


def test_none_value_handling(model):
    """Test that the OllamaModel.__call__ method handles None values gracefully"""

    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns None
            return None

    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we still get a string response
    assert isinstance(response, str)
    assert response == "None"  # str(None) = "None"