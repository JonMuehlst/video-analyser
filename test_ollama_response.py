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
        """Process messages and return a response from the Ollama model.

        Args:
            messages: Either a string prompt or a list of message dictionaries
            **kwargs: Additional parameters like max_tokens, temperature, etc.

        Returns:
            str: The model's response text
        """
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
                        role = msg.get("role", "user")
                        content = msg.get("content", "")

                        # Handle content that is a list (like Anthropic format)
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    text_parts.append(item["text"])
                                elif isinstance(item, dict) and "type" in item and item["type"] == "text":
                                    text_parts.append(item.get("text", ""))

                            formatted_messages.append({
                                "role": role if role in ["user", "assistant", "system", "tool"] else "user",
                                "content": " ".join(text_parts)
                            })
                        else:
                            # Ensure content is a string
                            formatted_messages.append({
                                "role": role if role in ["user", "assistant", "system", "tool"] else "user",
                                "content": str(content)
                            })
                    else:
                        logger.warning(f"Skipping invalid message format: {msg}")
            else:
                # Handle other types of input by converting to string
                formatted_messages.append({"role": "user", "content": str(messages)})

            # Extract relevant parameters
            max_tokens = kwargs.get("max_tokens", 4096)
            temperature = kwargs.get("temperature", 0.7)
            stop_sequences = kwargs.get("stop_sequences", None)

            options = {
                "num_predict": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            if stop_sequences:
                options["stop"] = stop_sequences

            # Log what we're about to send
            logger.debug(f"Calling Ollama API with model: {self.model_name}")
            logger.debug(f"Number of messages: {len(formatted_messages)}")

            # Call Ollama API
            response = self.client.chat(
                model=self.model_name,
                messages=formatted_messages,
                options=options
            )

            # Log raw response type for debugging
            logger.debug(f"Raw Ollama response type: {type(response)}")
            if isinstance(response, str):
                logger.debug(f"String response from Ollama (first 100 chars): {response[:100]}")
            else:
                logger.debug(f"Non-string response from Ollama: {str(response)[:100]}")

            # Extract content from response with robust error handling
            # IMPORTANT: Always check for string first
            if isinstance(response, str):
                return response

            # Handle dictionary response
            if isinstance(response, dict):
                if 'message' in response:
                    message = response['message']
                    if isinstance(message, dict) and 'content' in message:
                        return str(message['content'])
                    # Handle case where message itself might be a string or other type
                    return str(message)
                elif 'content' in response:
                    return str(response['content'])
                elif 'response' in response:
                    return str(response['response'])

            # Handle object with attributes
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

            # Last resort: convert to string
            return str(response)

        except Exception as e:
            logger.error(f"Error in OllamaModel.__call__: {str(e)}")
            return f"Error calling Ollama API: {str(e)}"


# Copy the full __call__ method from above here

def test_string_response_handling():
    """Test that the OllamaModel.__call__ method correctly handles string responses"""

    # Mock the OllamaModel class
    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns a string directly
            return "This is a direct string response from Ollama"

    # Create a model instance with our mock client
    model = OllamaModel(model_name="test-model")
    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the string response correctly
    assert response == "This is a direct string response from Ollama"
    assert isinstance(response, str)

    print("✅ String response handled correctly")


def test_object_response_handling():
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

    # Create a model instance with our mock client
    model = OllamaModel(model_name="test-model")
    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the response correctly extracted from the object structure
    assert response == "Response from message.content attribute"
    assert isinstance(response, str)

    print("✅ Object response handled correctly")


def test_dict_response_handling():
    """Test that the OllamaModel.__call__ method correctly handles dictionary responses"""

    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where Ollama returns a dictionary
            return {
                "message": {
                    "content": "Response from message.content in dictionary"
                }
            }

    # Create a model instance with our mock client
    model = OllamaModel(model_name="test-model")
    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got the response correctly extracted from the dictionary
    assert response == "Response from message.content in dictionary"
    assert isinstance(response, str)

    print("✅ Dictionary response handled correctly")


def test_error_case_handling():
    """Test that the OllamaModel.__call__ method handles errors gracefully"""

    class MockClient:
        def chat(self, model, messages, options=None):
            # Simulate the case where accessing message.content raises an AttributeError
            class BrokenResponse:
                @property
                def message(self):
                    # This property will raise an AttributeError when accessed
                    raise AttributeError("This object has no attribute 'content'")

            return BrokenResponse()

    # Create a model instance with our mock client
    model = OllamaModel(model_name="test-model")
    model.client = MockClient()

    # Test with a simple message
    response = model("Hello, how are you?")

    # Check that we got an error message instead of a crash
    assert "Error" in response
    assert isinstance(response, str)

    print("✅ Error case handled correctly")


# Run tests
def run_tests():
    print("Running Ollama response handling tests...")
    test_string_response_handling()
    test_object_response_handling()
    test_dict_response_handling()
    test_error_case_handling()
    print("All tests passed! ✅")


if __name__ == "__main__":
    run_tests()