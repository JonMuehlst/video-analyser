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
                try:
                    # Use LiteLLM for Ollama vision model
                    from smolagents import LiteLLMModel
                        
                    base_url = ollama_config.get("base_url", "http://localhost:11434")
                    vision_model = ollama_config.get("vision_model", "llava")
                        
                    # Check if Ollama is running
                    try:
                        import requests
                        response = requests.get(f"{base_url}/api/tags", timeout=5)
                        if response.status_code != 200:
                            return "Error: Cannot connect to Ollama. Please make sure Ollama is running with 'ollama serve'"
                            
                        # Check if the vision model is available
                        available_models = [m["name"] for m in response.json().get("models", [])]
                        if vision_model not in available_models:
                            return f"Error: Vision model '{vision_model}' not available in Ollama. Please pull it with 'ollama pull {vision_model}'"
                    except Exception as e:
                        return f"Error connecting to Ollama: {str(e)}"
                        
                    # Format model name for LiteLLM
                    litellm_vision_model = f"ollama/{vision_model}"
                        
                    # Create LiteLLM model for vision
                    litellm_model = LiteLLMModel(
                        model_id=litellm_vision_model,
                        api_base=base_url,
                        api_key="ollama",  # Placeholder, not used by Ollama
                        temperature=0.7,
                        max_tokens=4096,
                        request_timeout=120,  # Longer timeout for vision models
                    )
                        
                    # Determine optimal batch size based on model
                    # These are conservative defaults that should work on most GPUs
                    max_images_per_request = 3  # Default for 12GB VRAM
                        
                    # Adjust based on model size
                    if "tiny" in vision_model or "1b" in vision_model:
                        max_images_per_request = 6  # Smaller models can handle more images
                    elif "7b" in vision_model:
                        max_images_per_request = 3  # Medium models
                    else:
                        max_images_per_request = 2  # Larger models need smaller batches
                        
                    # Process in batches
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
                                
                            try:
                                # Format messages for vision model with images
                                messages = self._format_vision_messages(sub_prompt, images)
                                    
                                # Call the LiteLLM model
                                sub_result = litellm_model(messages)
                                sub_batch_results.append(sub_result)
                                    
                                # Log progress
                                logger.info(f"Processed sub-batch {i//max_images_per_request + 1}/{(len(batch) + max_images_per_request - 1)//max_images_per_request}")
                                    
                            except Exception as e:
                                error_msg = f"Error processing sub-batch: {str(e)}"
                                logger.error(error_msg)
                                sub_batch_results.append(f"Error in batch {i//max_images_per_request + 1}: {error_msg}")
                            
                        # Combine results
                        analysis = "\n\n".join(sub_batch_results)
                    else:
                        # Process the whole batch at once
                        # Extract base64 images
                        images = []
                        for frame in batch:
                            base64_image = frame.get('base64_image', '')
                            if base64_image and isinstance(base64_image, str):
                                images.append(base64_image)
                            
                        # Format messages for vision model with images
                        messages = self._format_vision_messages(prompt, images)
                            
                        # Call the LiteLLM model
                        analysis = litellm_model(messages)
                except ImportError:
                    return "Error: Required package 'requests' is not installed. Please install it with 'pip install requests'"
                except Exception as e:
                    error_msg = f"Error using Ollama for vision analysis: {str(e)}"
                    logger.error(error_msg)
                    return error_msg
                
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

    def _format_vision_messages(self, prompt: str, images: List[str]) -> List[Dict]:
        """Format messages for vision models with images in LiteLLM format"""
        # Create a list with a single message containing text and images
        message_content = [{"type": "text", "text": prompt}]
        
        # Add images to the content
        for base64_image in images:
            if base64_image and isinstance(base64_image, str):
                # Ensure proper formatting with data URI if needed
                if not base64_image.startswith("data:"):
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                else:
                    image_url = base64_image
                    
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
        
        # Return as a properly formatted message list
        return [{"role": "user", "content": message_content}]
    
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
