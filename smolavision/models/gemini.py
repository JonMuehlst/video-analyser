import logging
import base64
from typing import List, Dict, Any, Optional

import litellm
from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

# Set LiteLLM logger level if desired (e.g., to WARNING to reduce verbosity)
# litellm.set_verbose = False
# logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class GeminiModel(ModelInterface):
    """Implementation for Google Gemini models via LiteLLM."""

    DEFAULT_MODEL = "gemini/gemini-1.5-flash-latest" # Default to a generally available model
    DEFAULT_VISION_MODEL = "gemini/gemini-1.5-flash-latest" # Vision model

    def __init__(self,
                 api_key: str,
                 model_id: str = "gemini-1.5-flash-latest", # Model name without 'gemini/' prefix
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 **kwargs):
        """
        Initialize Gemini model via LiteLLM.

        Args:
            api_key: Google AI Studio API key (or GCP credentials if using Vertex)
            model_id: Gemini model ID (e.g., "gemini-1.5-pro-latest", "gemini-1.5-flash-latest")
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for text generation
            **kwargs: Additional parameters to pass to LiteLLM
        """
        if not api_key:
             raise ModelError("GEMINI_API_KEY is required for GeminiModel.")

        self.api_key = api_key
        # Ensure model_id is prefixed correctly for LiteLLM
        self.model_id = model_id if model_id.startswith("gemini/") else f"gemini/{model_id}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs # Store extra kwargs for LiteLLM
        logger.info(f"Initialized Gemini model via LiteLLM: {self.model_id}")

    def _prepare_litellm_kwargs(self, **call_kwargs) -> Dict[str, Any]:
        """Merge instance kwargs with call-specific kwargs for LiteLLM."""
        merged_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs, # Instance level kwargs
            **call_kwargs   # Call level kwargs
        }
        # Remove None values as LiteLLM might not handle them for all params
        return {k: v for k, v in merged_kwargs.items() if v is not None}

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using Gemini via LiteLLM."""
        try:
            messages = [{"role": "user", "content": prompt}]
            litellm_kwargs = self._prepare_litellm_kwargs(**kwargs)

            logger.debug(f"Calling LiteLLM completion for text generation with model {self.model_id}")
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                api_key=self.api_key,
                **litellm_kwargs
            )
            # Access content safely
            content = response.choices[0].message.content
            logger.debug(f"LiteLLM text generation successful, response length: {len(content or '')}")
            return content or ""
        except Exception as e:
            logger.exception(f"LiteLLM text generation failed for model {self.model_id}")
            # Attempt to get more specific error info from LiteLLM exception if possible
            error_message = str(getattr(e, 'message', e))
            raise ModelError(f"Gemini text generation via LiteLLM failed: {error_message}") from e

    def analyze_images(self, images: List[str], prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Analyze images with a text prompt using Gemini vision models via LiteLLM."""
        # Determine the correct vision model ID to use
        # Use the main model_id if it's a known vision model, otherwise default
        vision_model_id = self.model_id
        if "flash" not in vision_model_id and "pro" not in vision_model_id: # Simple check, might need refinement
             logger.warning(f"Model {self.model_id} might not support vision. Using default vision model.")
             vision_model_id = self.DEFAULT_VISION_MODEL

        # Override max_tokens if provided specifically for this call
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Construct messages for multimodal input according to LiteLLM's expected format
            content = [{"type": "text", "text": prompt}]
            for image_data in images:
                # LiteLLM expects base64 string directly for gemini/
                content.append({
                    "type": "image_url",
                    "image_url": {
                        # Assuming JPEG format, adjust if necessary
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })

            messages = [{"role": "user", "content": content}]
            litellm_kwargs = self._prepare_litellm_kwargs(max_tokens=current_max_tokens, **kwargs)

            logger.debug(f"Calling LiteLLM completion for image analysis with model {vision_model_id}")
            response = litellm.completion(
                model=vision_model_id,
                messages=messages,
                api_key=self.api_key,
                **litellm_kwargs
            )
            # Access content safely
            content = response.choices[0].message.content
            logger.debug(f"LiteLLM image analysis successful, response length: {len(content or '')}")
            return content or ""
        except Exception as e:
            logger.exception(f"LiteLLM image analysis failed for model {vision_model_id}")
            # Attempt to get more specific error info from LiteLLM exception if possible
            error_message = str(getattr(e, 'message', e))
            raise ModelError(f"Gemini image analysis via LiteLLM failed: {error_message}") from e
