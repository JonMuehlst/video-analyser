import pytest
from unittest.mock import patch, MagicMock
import base64
import json

from smolavision.models.anthropic import AnthropicModel
from smolavision.models.openai import OpenAIModel
from smolavision.models.ollama import OllamaModel
from smolavision.models.huggingface import HuggingFaceModel
from smolavision.models.gemini import GeminiModel # Import GeminiModel
from smolavision.models.factory import ModelFactory
from smolavision.exceptions import ModelError, ConfigurationError


class TestAnthropicModel:
    """Tests for the Anthropic model implementation."""

    @patch('anthropic.Anthropic')
    def test_generate_text(self, mock_anthropic_class):
        """Test text generation with Anthropic model."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated text")]
        mock_client.messages.create.return_value = mock_response
        
        # Create model and call generate_text
        model = AnthropicModel(api_key="test_key")
        result = model.generate_text("Test prompt")
        
        # Verify results
        assert result == "Generated text"
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_analyze_images(self, mock_anthropic_class):
        """Test image analysis with Anthropic model."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Image analysis")]
        mock_client.messages.create.return_value = mock_response
        
        # Create model and call analyze_images
        model = AnthropicModel(api_key="test_key")
        result = model.analyze_images(
            images=["base64image1", "base64image2"],
            prompt="Analyze these images"
        )
        
        # Verify results
        assert result == "Image analysis"
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_error_handling(self, mock_anthropic_class):
        """Test error handling in Anthropic model."""
        # Mock Anthropic client to raise an exception
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")
        
        # Create model and call generate_text
        model = AnthropicModel(api_key="test_key")
        
        # Verify exception is raised
        with pytest.raises(ModelError):
            model.generate_text("Test prompt")


class TestOpenAIModel:
    """Tests for the OpenAI model implementation."""

    @patch('openai.OpenAI')
    def test_generate_text(self, mock_openai_class):
        """Test text generation with OpenAI model."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create model and call generate_text
        model = OpenAIModel(api_key="test_key")
        result = model.generate_text("Test prompt")
        
        # Verify results
        assert result == "Generated text"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_analyze_images(self, mock_openai_class):
        """Test image analysis with OpenAI model."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Image analysis"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create model and call analyze_images
        model = OpenAIModel(api_key="test_key")
        result = model.analyze_images(
            images=["base64image1", "base64image2"],
            prompt="Analyze these images"
        )
        
        # Verify results
        assert result == "Image analysis"
        mock_client.chat.completions.create.assert_called_once()


class TestGeminiModel:
    """Tests for the Gemini model implementation via LiteLLM."""

    @patch('litellm.completion')
    def test_generate_text(self, mock_litellm_completion):
        """Test text generation with Gemini model via LiteLLM."""
        # Mock LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
        mock_litellm_completion.return_value = mock_response

        # Create model and call generate_text
        model = GeminiModel(api_key="test_key", model_id="gemini-1.5-flash-latest")
        result = model.generate_text("Test prompt")

        # Verify results
        assert result == "Generated text"
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": "Test prompt"}],
            api_key="test_key",
            temperature=0.7, # Default from GeminiModel
            max_tokens=4096  # Default from GeminiModel
        )

    @patch('litellm.completion')
    def test_analyze_images(self, mock_litellm_completion):
        """Test image analysis with Gemini model via LiteLLM."""
        # Mock LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Image analysis"))]
        mock_litellm_completion.return_value = mock_response

        # Create model and call analyze_images
        model = GeminiModel(api_key="test_key", model_id="gemini-1.5-flash-latest")
        images = ["base64image1", "base64image2"]
        prompt = "Analyze these images"
        result = model.analyze_images(images=images, prompt=prompt)

        # Verify results
        assert result == "Image analysis"
        expected_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64image1"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64image2"}},
            ]
        }]
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-1.5-flash-latest", # Should use vision model logic if applicable
            messages=expected_messages,
            api_key="test_key",
            temperature=0.7,
            max_tokens=4096 # Default from analyze_images signature
        )

    @patch('litellm.completion', side_effect=Exception("LiteLLM API error"))
    def test_error_handling(self, mock_litellm_completion):
        """Test error handling in Gemini model."""
        model = GeminiModel(api_key="test_key", model_id="gemini-1.5-flash-latest")
        with pytest.raises(ModelError, match="LiteLLM API error"):
            model.generate_text("Test prompt")


class TestOllamaModel:
    """Tests for the Ollama model implementation."""

    @patch('httpx.Client')
    def test_generate_text(self, mock_client_class):
        """Test text generation with Ollama model."""
        # Mock httpx client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_client.post.return_value = mock_response
        
        # Create model and call generate_text
        model = OllamaModel()
        result = model.generate_text("Test prompt")
        
        # Verify results
        assert result == "Generated text"
        mock_client.post.assert_called_once()

    @patch('httpx.Client')
    def test_analyze_images(self, mock_client_class):
        """Test image analysis with Ollama model."""
        # Mock httpx client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"content": "Image analysis"}]}
        mock_client.post.return_value = mock_response
        
        # Create model and call analyze_images
        model = OllamaModel()
        result = model.analyze_images(
            images=["base64image1", "base64image2"],
            prompt="Analyze these images"
        )
        
        # Verify results
        assert result == "Image analysis"
        mock_client.post.assert_called_once()


class TestModelFactory:
    """Tests for the model factory."""

    def test_create_anthropic_model(self):
        """Test creating an Anthropic model."""
        with patch('smolavision.models.factory.AnthropicModel') as mock_model_class:
            config = {
                "model_type": "anthropic",
                "api_key": "test_key",
                "vision_model": "claude-3-opus-20240229"
            }
            
            ModelFactory.create_vision_model(config)
            
            mock_model_class.assert_called_once_with(
                api_key="test_key",
                model_id="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=4096
            )

    def test_create_openai_model(self):
        """Test creating an OpenAI model."""
        with patch('smolavision.models.factory.OpenAIModel') as mock_model_class:
            config = {
                "model_type": "openai",
                "api_key": "test_key",
                "vision_model": "gpt-4o"
            }
            
            ModelFactory.create_vision_model(config)
            
            mock_model_class.assert_called_once_with(
                api_key="test_key",
                model_id="gpt-4o",
                temperature=0.7,
                max_tokens=4096
            )

    def test_create_ollama_model(self):
        """Test creating an Ollama model."""
        with patch('smolavision.models.factory.OllamaModel') as mock_model_class:
            config = {
                "model_type": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model_name": "llama3",
                    "vision_model": "llava"
                }
            }
            
            ModelFactory.create_vision_model(config)
            
            mock_model_class.assert_called_once()

    def test_create_gemini_model(self):
        """Test creating a Gemini model."""
        with patch('smolavision.models.factory.GeminiModel') as mock_model_class, \
             patch.dict(os.environ, {"GEMINI_API_KEY": "env_key"}, clear=True): # Mock env var
            config = {
                "model_type": "gemini",
                # api_key can be omitted if env var is set
                "model_name": "gemini-1.5-flash-latest" # Name without prefix
            }

            ModelFactory.create_model(config) # Use create_model directly

            mock_model_class.assert_called_once_with(
                api_key="env_key", # Factory should retrieve from env
                model_id="gemini-1.5-flash-latest",
                temperature=0.7,
                max_tokens=4096
            )

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        config = {
            "model_type": "invalid",
            "api_key": "test_key"
        }
        
        with pytest.raises(ConfigurationError):
            ModelFactory.create_vision_model(config)

    def test_missing_api_key(self):
        """Test error handling for missing API key."""
        # Test missing API key for Gemini
        with patch.dict(os.environ, {}, clear=True): # Ensure env var is not set
            config = {
                "model_type": "gemini"
            }
            with pytest.raises(ConfigurationError, match="API key is required for gemini models"):
                ModelFactory.create_model(config)
