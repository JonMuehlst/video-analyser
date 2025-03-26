"""
Test LiteLLM integration with Ollama
"""
import pytest
import logging
import os
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiteLLMOllamaTest")

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

@pytest.mark.skipif(not check_ollama_server(), reason="Ollama server not running")
def test_litellm_ollama_text():
    """Test LiteLLM with Ollama for text generation"""
    try:
        from smolagents import LiteLLMModel
        
        # Use a small, fast model for testing
        model = LiteLLMModel(
            model_id="ollama/phi3:mini",
            api_base="http://localhost:11434",
            api_key="ollama",  # Placeholder, not used
            max_tokens=100,  # Small for faster tests
            temperature=0.7,
        )
        
        # Test basic functionality
        prompt = "Hello, how are you?"
        messages = [{"role": "user", "content": prompt}]
        
        response = model(messages)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        logger.info(f"Response: {response[:100]}...")
        
    except ImportError:
        pytest.skip("LiteLLM not installed")
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise

@pytest.mark.skipif(not check_ollama_server(), reason="Ollama server not running")
def test_litellm_ollama_vision():
    """Test LiteLLM with Ollama for vision tasks"""
    try:
        import base64
        from smolagents import LiteLLMModel
        from pathlib import Path
        
        # Skip if no vision models available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            models = [m["name"] for m in response.json().get("models", [])]
            vision_models = [m for m in models if "llava" in m.lower()]
            if not vision_models:
                pytest.skip("No vision models available in Ollama")
            vision_model = vision_models[0]
        except Exception:
            pytest.skip("Could not check for vision models")
        
        # Create a test image (or use a placeholder)
        test_image_path = Path(__file__).parent / "test_image.jpg"
        if not test_image_path.exists():
            # Create a simple test image or skip
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color = (73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((10,10), "Test", fill=(255,255,0))
                img.save(test_image_path)
            except ImportError:
                pytest.skip("PIL not installed for image creation")
        
        # Load and encode the image
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create LiteLLM model for vision
        model = LiteLLMModel(
            model_id=f"ollama/{vision_model}",
            api_base="http://localhost:11434",
            api_key="ollama",  # Placeholder, not used
            max_tokens=100,  # Small for faster tests
            temperature=0.7,
        )
        
        # Create message with image
        message_content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        
        messages = [{"role": "user", "content": message_content}]
        
        # Call the model
        response = model(messages)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        logger.info(f"Vision response: {response[:100]}...")
        
    except ImportError:
        pytest.skip("Required libraries not installed")
    except Exception as e:
        logger.error(f"Error in vision test: {str(e)}")
        raise

if __name__ == "__main__":
    # Run tests directly if Ollama is available
    if check_ollama_server():
        test_litellm_ollama_text()
        test_litellm_ollama_vision()
    else:
        print("Ollama server not running, skipping tests")
