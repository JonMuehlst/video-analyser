#!/usr/bin/env python
"""
Vision analysis tool for SmolaVision
"""

import os
import logging
from typing import List, Dict, Any, Optional

from smolagents import Tool

# Configure logging
logger = logging.getLogger("SmolaVision")


class VisionAnalysisTool(Tool):
    """Tool for analyzing batches of frames using a vision model"""
    name = "analyze_batch"
    description = "Analyzes a batch of frames using a vision model"
    inputs = {
        "batch": {
            "type": "array",
            "description": "Batch of frames to analyze"
        },
        "previous_context": {
            "type": "string",
            "description": "Context from previous analysis for continuity",
            "nullable": True
        },
        "language": {
            "type": "string",
            "description": "Language of text in the video",
            "nullable": True
        },
        "model_name": {
            "type": "string",
            "description": "Name of the vision model to use (claude or gpt4o)",
            "nullable": True
        },
        "api_key": {
            "type": "string",
            "description": "API key for the vision model",
            "nullable": True
        },
        "mission": {
            "type": "string",
            "description": "Specific analysis mission (e.g., 'workflow', 'general')",
            "nullable": True
        },
        "ollama_config": {
            "type": "object",
            "description": "Configuration for Ollama local models",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, batch: List[Dict],
                previous_context: str = "",
                language: str = "Hebrew",
                model_name: str = "claude",
                api_key: str = "",
                mission: str = "general",
                ollama_config: Optional[Dict] = None) -> str:
        """Analyze a batch of frames using a vision model"""
        if not batch or len(batch) == 0:
            return "Error: Empty batch received"

        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Ensure model_name is not None
        if model_name is None:
            model_name = "claude"
            logger.warning("model_name was None, defaulting to 'claude'")

        # Get first and last timestamp for logging
        start_time = batch[0].get("timestamp", "unknown")
        end_time = batch[-1].get("timestamp", "unknown")

        logger.info(f"Analyzing batch from {start_time} to {end_time} ({len(batch)} frames)")

        try:
            # Create prompt based on mission type
            if mission and mission.lower() == "workflow":
                prompt = self._create_workflow_prompt(batch, previous_context, language)
            else:
                prompt = self._create_general_prompt(batch, previous_context, language)

            # Check if using Ollama
            if model_name == "ollama" or (ollama_config and ollama_config.get("enabled")):
                # Use Ollama for local inference
                from ollama_client import OllamaClient
                
                base_url = ollama_config.get("base_url", "http://localhost:11434")
                vision_model = ollama_config.get("vision_model", "llava")
                
                # Create or reuse Ollama client
                if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                    self._ollama_client = OllamaClient(base_url=base_url)
                
                # For smaller GPUs, we need to be careful with batch size
                # Process images in smaller sub-batches if needed
                max_images_per_request = 3  # Limit for 12GB VRAM
                
                if len(batch) > max_images_per_request:
                    logger.info(f"Batch size ({len(batch)}) exceeds max images per request ({max_images_per_request}). Processing in sub-batches.")
                    
                    # Process in sub-batches and combine results
                    sub_batch_results = []
                    for i in range(0, len(batch), max_images_per_request):
                        sub_batch = batch[i:i + max_images_per_request]
                        
                        # Extract base64 images for this sub-batch
                        images = []
                        for frame in sub_batch:
                            base64_image = frame.get('base64_image', '')
                            if base64_image and isinstance(base64_image, str):
                                images.append(base64_image)
                        
                        # Create a sub-prompt
                        sub_prompt = f"Analyzing frames {i+1} to {i+len(sub_batch)} of {len(batch)}:\n{prompt}"
                        
                        # Call Ollama vision model for this sub-batch
                        sub_result = self._ollama_client.generate_vision(
                            model=vision_model,
                            prompt=sub_prompt,
                            images=images,
                            max_tokens=2048  # Smaller token limit for sub-batches
                        )
                        
                        sub_batch_results.append(sub_result)
                    
                    # Combine results
                    analysis = "\n\n".join(sub_batch_results)
                else:
                    # Process the whole batch at once
                    # Extract base64 images for Ollama
                    images = []
                    for frame in batch:
                        base64_image = frame.get('base64_image', '')
                        if base64_image and isinstance(base64_image, str):
                            images.append(base64_image)
                    
                    # Call Ollama vision model
                    analysis = self._ollama_client.generate_vision(
                        model=vision_model,
                        prompt=prompt,
                        images=images,
                        max_tokens=4096
                    )
                
            # Prepare images based on the model
            elif model_name.startswith("claude"):
                # Anthropic Claude format
                import anthropic

                # Create client (or reuse cached one)
                if not hasattr(self, '_anthropic_client') or self._anthropic_client is None:
                    self._anthropic_client = anthropic.Anthropic(api_key=api_key)

                # Format images for Claude
                images = []
                for frame in batch:
                    base64_image = frame.get('base64_image', '')
                    if base64_image and isinstance(base64_image, str):
                        images.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        })

                # Make API call
                model_id = "claude-3-opus-20240229" if model_name == "claude" else model_name
                response = self._anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *images
                            ]
                        }
                    ]
                )
                analysis = response.content[0].text

            elif model_name == "gpt4o" or model_name == "gpt-4o":
                # OpenAI GPT-4 Vision format
                import openai

                # Create client (or reuse cached one)
                if not hasattr(self, '_openai_client') or self._openai_client is None:
                    self._openai_client = openai.OpenAI(api_key=api_key)

                # Format images for OpenAI
                images = []
                for frame in batch:
                    base64_image = frame.get('base64_image', '')
                    if base64_image and isinstance(base64_image, str):
                        images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })

                # Make API call
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *images
                            ]
                        }
                    ]
                )
                analysis = response.choices[0].message.content

            else:
                return f"Error: Unsupported model name: {model_name}. Use 'claude', 'gpt4o', or 'ollama'."

            logger.info(f"Successfully analyzed batch from {start_time} to {end_time}")

            # Save analysis to a file
            output_dir = os.path.join("output", "batch_analyses")
            os.makedirs(output_dir, exist_ok=True)

            safe_start = batch[0].get("safe_timestamp", "unknown")
            safe_end = batch[-1].get("safe_timestamp", "unknown")

            analysis_file = os.path.join(output_dir, f"batch_{safe_start}_to_{safe_end}.txt")
            with open(analysis_file, "w", encoding="utf-8") as f:
                # Add batch metadata as a header
                f.write(f"# Batch Analysis: {start_time} to {end_time}\n")
                f.write(f"# Frames: {len(batch)}\n")
                f.write(f"# Mission: {mission}\n")
                f.write(
                    f"# Scene changes: {len([frame for frame in batch if frame.get('is_scene_change', False)])}\n\n")
                f.write(analysis)

            return analysis

        except Exception as e:
            error_msg = f"Error analyzing batch from {start_time} to {end_time}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _create_general_prompt(self, batch, previous_context, language):
        """Create the standard general analysis prompt"""
        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Prepare timestamps, noting scene changes
        timestamp_info = []
        for frame in batch:
            if frame.get("is_scene_change", False):
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}*]")  # Mark scene changes with asterisk
            else:
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}]")

        timestamps_str = ", ".join(timestamp_info)

        # Add indicator for scene changes if any exist in this batch
        scene_changes = [frame for frame in batch if frame.get("is_scene_change", False)]
        scene_change_note = ""
        if scene_changes:
            scene_change_timestamps = [f"[{frame.get('timestamp', 'unknown')}]" for frame in scene_changes]
            scene_change_note = (f"\nNOTE: Frames marked with * represent detected scene changes at "
                                 f"{', '.join(scene_change_timestamps)}. Pay special attention to changes "
                                 f"in content, setting, or context at these points.")

        # Include OCR text if available
        ocr_texts = []
        for frame in batch:
            if frame.get("ocr_text"):
                ocr_texts.append(f"[{frame.get('timestamp', 'unknown')}]: {frame.get('ocr_text')}")

        ocr_section = ""
        if ocr_texts:
            ocr_section = "\n\nExtracted text from OCR:\n" + "\n".join(ocr_texts)

        # Prepare context
        if previous_context:
            context_intro = f"\n\nContext from previous video segments:\n{previous_context}\n\n"
        else:
            context_intro = ""

        # Create prompt
        prompt = f"""
{context_intro}
Analyze the following video frames captured at timestamps {timestamps_str}. The video is in {language} language.{scene_change_note}{ocr_section}

Key instructions:
1. Describe all visual elements, text, people, actions, and settings
2. Transcribe and translate any {language} text visible in the frames
3. Note any transitions or changes across frames, especially at scene change points
4. Maintain continuity with the previous context provided (if any)
5. If text appears to continue from previous frames or context, attempt to complete the meaning

Organize your analysis chronologically, noting timestamps where appropriate.
Be detailed and comprehensive in your analysis.
"""
        return prompt

    def _create_workflow_prompt(self, batch, previous_context, language):
        """Create a specialized prompt for workflow analysis"""
        # Ensure language is not None
        if language is None:
            language = "English"
            logger.warning("Language parameter was None, defaulting to English")

        # Prepare timestamps, noting scene changes
        timestamp_info = []
        for frame in batch:
            if frame.get("is_scene_change", False):
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}*]")  # Mark scene changes with asterisk
            else:
                timestamp_info.append(f"[{frame.get('timestamp', 'unknown')}]")

        timestamps_str = ", ".join(timestamp_info)

        # Check for scene changes
        scene_changes = [frame for frame in batch if frame.get("is_scene_change", False)]
        scene_change_note = ""
        if scene_changes:
            scene_change_timestamps = [f"[{frame.get('timestamp', 'unknown')}]" for frame in scene_changes]
            scene_change_note = (f"\nNOTE: Frames marked with * represent detected scene changes at "
                                 f"{', '.join(scene_change_timestamps)}. Pay special attention to workflow transitions at these points.")

        # Include OCR text if available
        ocr_texts = []
        for frame in batch:
            if frame.get("ocr_text"):
                ocr_texts.append(f"[{frame.get('timestamp', 'unknown')}]: {frame.get('ocr_text')}")

        ocr_section = ""
        if ocr_texts:
            ocr_section = "\n\nExtracted text from OCR:\n" + "\n".join(ocr_texts)

        # Prepare context
        if previous_context:
            context_intro = f"\n\nContext from previous video segments:\n{previous_context}\n\n"
        else:
            context_intro = ""

        # Create workflow-specific prompt
        prompt = f"""
{context_intro}
Analyze the following video frames captured at timestamps {timestamps_str}. These frames show interactions with an AI platform.{scene_change_note}{ocr_section}

Key instructions for workflow analysis:
1. Identify user roles and AI roles in the interaction (who is doing what)
2. Analyze the prompting patterns and styles used with the AI
3. Identify key steps in the workflow of interacting with the AI platform
4. Note the sequence of actions, inputs, and outputs
5. Identify any UI elements and their functions in the workflow
6. Note any specific "prompt engineering" techniques visible
7. Maintain continuity with the previous context provided (if any)
8. If text appears to continue from previous frames or context, attempt to complete the meaning

Organize your analysis chronologically, focusing on the workflow logic and interaction patterns.
Your analysis will later be used to create a flow diagram of the entire process.
"""
        return prompt
