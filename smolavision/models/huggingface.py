# smolavision/models/huggingface.py
import logging
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import pipeline
from smolavision.models.base import ModelInterface
from smolavision.exceptions import ModelError

logger = logging.getLogger(__name__)

class HuggingFaceModel(ModelInterface):
    """Implementation for Hugging Face Hub models."""

    def __init__(self, token: str = None, model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
                 text_pipeline_task: str = "text-generation", vision_pipeline_task: str = "image-to-text",
                 max_new_tokens: int = 4096, temperature: float = 0.7):
        """
        Initialize Hugging Face model.

        Args:
            token (optional): Hugging Face API token.
            model_id (optional): Hugging Face model ID (default: meta-llama/Llama-3.3-70B-Instruct).
            text_pipeline_task (optional): The task for the text generation pipeline.
            vision_pipeline_task (optional): The task for the vision pipeline.
            max_new_tokens: The maximum number of tokens for generation.
            temperature: The temperature for text generation
        """
        self.model_id = model_id
        self.text_pipeline_task = text_pipeline_task
        self.vision_pipeline_task = vision_pipeline_task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.text_pipeline = None
        self.vision_pipeline = None
        try:
            self.text_pipeline = pipeline(self.text_pipeline_task, model=self.model_id, token=token, device_map="auto")
        except Exception as e:
            logger.warning(f"Failed to initialize Hugging Face text generation pipeline: {e}")

        try:
            self.vision_pipeline = pipeline(self.vision_pipeline_task, model=self.model_id, token=token, device_map="auto")
        except Exception as e:
            logger.warning(f"Failed to initialize Hugging Face vision pipeline (task {self.vision_pipeline_task}), it may not be a visual model: {e}")

        logger.info(f"Initialized Hugging Face model: {self.model_id} (text task: {self.text_pipeline_task}, vision task: {self.vision_pipeline_task})")


    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using Hugging Face model."""
        if not self.text_pipeline:
            raise ModelError("Text generation pipeline not initialized. Check model ID and task.")
        try:
            # HF text generation pipeline expects the prompt as a direct argument
            result = self.text_pipeline(prompt, max_new_tokens=self.max_new_tokens, temperature = self.temperature, **kwargs)
            return result[0]['generated_text'] if result else ""
        except Exception as e:
            raise ModelError(f"Hugging Face text generation failed: {e}") from e

    def analyze_images(self, images: List[str], prompt: str, max_tokens: int = 4096, **kwargs) -> str:
        """Analyze images with a text prompt using Hugging Face model."""
        if not self.vision_pipeline:
            raise ModelError("Vision pipeline not initialized. Check model ID and task.")

        try:
            # Load images using PIL
            pil_images = []
            for image_data in images:
                decoded_image_data = base64.b64decode(image_data)
                pil_images.append(Image.open(BytesIO(decoded_image_data)))

            # HF image-to-text pipeline expects the image(s) as a direct argument
            result = self.vision_pipeline(pil_images, prompt=prompt, max_new_tokens=max_tokens, **kwargs)

            if isinstance(result, list):
                return "\n".join(res["generated_text"] for res in result if "generated_text" in res)
            elif isinstance(result, str):
                return result
            else:
                 return result[0]['generated_text'] if isinstance(result,list) and result[0] and "generated_text" in result[0]  else result if isinstance(result,str) else ""
        except Exception as e:
            raise ModelError(f"Hugging Face image analysis failed: {e}") from e