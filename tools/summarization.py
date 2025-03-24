#!/usr/bin/env python
"""
Summarization tool for SmolaVision
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional

from smolagents import Tool

# Configure logging
logger = logging.getLogger("SmolaVision")


class SummarizationTool(Tool):
    """Tool for generating a coherent summary from batch analyses"""
    name = "generate_summary"
    description = "Creates a coherent summary from all batch analyses"
    inputs = {
        "analyses": {
            "type": "array",
            "description": "List of all batch analyses"
        },
        "language": {
            "type": "string",
            "description": "Language of text in the video",
            "nullable": True
        },
        "model_name": {
            "type": "string",
            "description": "Name of the LLM to use for summarization",
            "nullable": True
        },
        "api_key": {
            "type": "string",
            "description": "API key for the LLM",
            "nullable": True
        },
        "mission": {
            "type": "string",
            "description": "Specific analysis mission (e.g., 'workflow', 'general')",
            "nullable": True
        },
        "generate_flowchart": {
            "type": "boolean",
            "description": "Whether to generate a flowchart diagram",
            "nullable": True
        },
        "ollama_config": {
            "type": "object",
            "description": "Configuration for Ollama local models",
            "nullable": True
        }
    }
    output_type = "object"

    def forward(self, analyses: List[str],
                language: str = "Hebrew",
                model_name: str = "claude-3-5-sonnet-20240620",
                api_key: str = "",
                mission: str = "general",
                generate_flowchart: bool = False,
                ollama_config: Optional[Dict] = None) -> Dict:
        """Generate a coherent summary from all batch analyses"""
        logger.info("Generating final summary from all analyses")

        # Check for empty or None analyses
        if not analyses:
            return {"error": "No analyses provided"}

        # Filter out None values
        analyses = [analysis for analysis in analyses if analysis is not None]
        if not analyses:
            return {"error": "All analyses were None"}

        try:
            # Ensure language is not None
            if language is None:
                language = "English"
                logger.warning("Language parameter was None, defaulting to English")

            # Ensure model_name is not None
            if model_name is None:
                model_name = "claude-3-5-sonnet-20240620"
                logger.warning("model_name was None, defaulting to default model")

            # First, concatenate all analyses for the full detailed version
            full_analysis = "\n\n---\n\n".join(analyses)

            # Save the full analysis to a file
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            full_analysis_path = os.path.join(output_dir, "video_analysis_full.txt")
            with open(full_analysis_path, "w", encoding="utf-8") as f:
                f.write(full_analysis)

            logger.info(f"Saved full analysis to {full_analysis_path}")

            # Generate a coherent summary using the specified LLM
            logger.info(f"Generating coherent narrative summary using {model_name}")

            # Split the full analysis into manageable chunks for the LLM
            max_chunk_size = 12000
            chunks = self._chunk_text(full_analysis, max_chunk_size)
            logger.info(f"Split analysis into {len(chunks)} chunks for summarization")

            # Choose prompt template based on mission type and flowchart request
            if mission and mission.lower() == "workflow" and generate_flowchart:
                prompt_template = self._workflow_with_flowchart_prompt_template
            elif mission and mission.lower() == "workflow":
                prompt_template = self._workflow_prompt_template
            else:
                prompt_template = self._general_prompt_template

            # Process each chunk to build a complete summary
            complete_summary = ""

            for i, chunk in enumerate(chunks):
                is_first = (i == 0)
                is_last = (i == len(chunks) - 1)

                logger.info(f"Generating summary for chunk {i + 1}/{len(chunks)}")

                # Create a prompt based on chunk position
                if is_first and is_last:  # Only one chunk
                    prompt = prompt_template.format(language=language, chunk=chunk)
                elif is_first:  # First of multiple chunks
                    prompt = f"""
You are analyzing a video in {language}. Below is the first part of a detailed analysis of the video frames.

Please begin creating a well-structured summary. This is part 1 of a multi-part summary process.
Focus on:
1. Describing the key visual elements, settings, people, and actions
2. Including all important {language} text with translations
3. Maintaining chronological flow
4. Setting up context for later parts of the video

The goal is to start a cohesive narrative that will be continued with additional content.

VIDEO ANALYSIS (PART 1):
{chunk}
"""
                elif is_last:  # Last of multiple chunks
                    prompt = f"""
You are continuing to analyze a video in {language}. Below is the final part of a detailed analysis of the video frames.

This is the final part in a multi-part summary process. You've already summarized earlier parts as follows:

PREVIOUS SUMMARY:
{complete_summary}

Please complete the summary by:
1. Continuing the narrative from where the previous summary left off
2. Integrating new information from this final section
3. Ensuring all important {language} text is included with translations
4. Creating proper closure and concluding the summary
5. Maintaining consistency with the style and approach of the previous summary

VIDEO ANALYSIS (FINAL PART):
{chunk}
"""
                    # If this is workflow with flowchart and we're on the last chunk, add flowchart instructions
                    if mission and mission.lower() == "workflow" and generate_flowchart:
                        prompt += """
After completing the narrative summary, please create a section titled "Workflow Diagram" containing a Mermaid flowchart that visualizes the workflow.
Use this format for the Mermaid diagram:
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
    %% Add all needed nodes and connections to represent the workflow
```
"""
                else:  # Middle chunk
                    prompt = f"""
You are continuing to analyze a video in {language}. Below is a middle part of a detailed analysis of the video frames.

This is a continuation of a multi-part summary process. You've already summarized earlier parts as follows:

PREVIOUS SUMMARY:
{complete_summary}

Please continue the summary by:
1. Picking up where the previous summary left off
2. Integrating new information from this section
3. Ensuring all important {language} text is included with translations
4. Maintaining chronological flow and narrative coherence
5. Setting up context for later parts of the video

VIDEO ANALYSIS (MIDDLE PART):
{chunk}
"""

                # Check if using Ollama
                if model_name == "ollama" or (ollama_config and ollama_config.get("enabled")):
                    # Use Ollama for local inference
                    from ollama_client import OllamaClient
                    
                    base_url = ollama_config.get("base_url", "http://localhost:11434")
                    model = ollama_config.get("model_name", "llama3")
                    
                    # Create or reuse Ollama client
                    if not hasattr(self, '_ollama_client') or self._ollama_client is None:
                        self._ollama_client = OllamaClient(base_url=base_url)
                    
                    # For smaller models, we need to be more careful with context length
                    # Determine if we should use a smaller model for better performance
                    if len(prompt) > 8000 and "small_models" in ollama_config:
                        # Use the smallest model for very large prompts
                        logger.info(f"Using smaller model for large prompt ({len(prompt)} chars)")
                        model = ollama_config.get("small_models", {}).get("fast", model)
                    
                    # Call Ollama model with appropriate token limit
                    # Smaller models need smaller token limits
                    max_tokens = 2048 if "phi" in model or "gemma:2b" in model or "tiny" in model else 4096
                    
                    chunk_summary = self._ollama_client.generate(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens
                    )
                
                # Call the LLM API based on model name
                elif model_name.startswith("claude"):
                    import anthropic

                    # Create or reuse client
                    if not hasattr(self, '_anthropic_client') or self._anthropic_client is None:
                        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

                    response = self._anthropic_client.messages.create(
                        model=model_name,
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    chunk_summary = response.content[0].text

                elif model_name.startswith("gpt"):
                    import openai

                    # Create or reuse client
                    if not hasattr(self, '_openai_client') or self._openai_client is None:
                        self._openai_client = openai.OpenAI(api_key=api_key)

                    response = self._openai_client.chat.completions.create(
                        model=model_name,
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    chunk_summary = response.choices[0].message.content

                else:
                    return {"error": f"Unsupported model: {model_name}"}

                # Update the complete summary
                if is_first:
                    complete_summary = chunk_summary
                else:
                    # For non-first chunks, append the new summary
                    complete_summary += "\n\n" + chunk_summary

            # If flowchart was requested and generated, save it
            flowchart_path = None
            if generate_flowchart and mission and mission.lower() == "workflow":
                # Extract the flowchart Mermaid code
                flowchart_pattern = r"```mermaid\s*([\s\S]*?)\s*```"
                flowchart_match = re.search(flowchart_pattern, complete_summary)

                if flowchart_match:
                    flowchart_code = flowchart_match.group(1).strip()
                    flowchart_path = os.path.join(output_dir, "workflow_flowchart.mmd")

                    with open(flowchart_path, "w", encoding="utf-8") as f:
                        f.write(flowchart_code)

                    logger.info(f"Saved workflow flowchart to {flowchart_path}")

            # Save the coherent summary to a file
            summary_path = os.path.join(output_dir, "video_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(complete_summary)

            logger.info(f"Saved coherent summary to {summary_path}")

            # Return results including flowchart if generated
            result = {
                "full_analysis": full_analysis_path,
                "coherent_summary": summary_path,
                "summary_text": complete_summary
            }

            if flowchart_path:
                result["flowchart"] = flowchart_path

            return result

        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _chunk_text(self, text: str, max_chunk_size: int = 12000) -> List[str]:
        """Split text into chunks for processing by LLM"""
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, start a new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Templates for different types of summaries
    _general_prompt_template = """
You are analyzing a video in {language}. Below is a detailed analysis of the video frames.

Please create a well-structured, comprehensive summary of the entire video. This summary should:
1. Describe the key visual elements, settings, people, and actions
2. Include all important {language} text with translations
3. Maintain chronological flow and narrative coherence
4. Highlight main topics or themes
5. Be detailed while eliminating redundancy

The goal is to create a cohesive narrative that someone could read to understand the full content of the video.

VIDEO ANALYSIS:
{chunk}
"""

    _workflow_prompt_template = """
You are analyzing a video that demonstrates workflow interactions with an AI platform. Below is a detailed analysis of the video frames.

Please create a well-structured, comprehensive summary of the AI interaction workflow shown in the video. This summary should:
1. Identify distinct roles in the interaction (user types, AI roles)
2. Describe the key steps in the workflow sequence
3. Analyze prompting patterns and strategies used
4. Describe UI elements and their functions in the workflow
5. Maintain chronological flow and logical sequence
6. Highlight important techniques or best practices demonstrated

The goal is to create a clear description of the entire workflow that someone could follow to understand or replicate the process.

VIDEO ANALYSIS:
{chunk}
"""

    _workflow_with_flowchart_prompt_template = """
You are analyzing a video that demonstrates workflow interactions with an AI platform. Below is a detailed analysis of the video frames.

Please create:
1. A well-structured, comprehensive summary of the AI interaction workflow shown in the video
2. A Mermaid flowchart diagram that visually represents the workflow

For the SUMMARY, include:
- Identification of distinct roles in the interaction (user types, AI roles)
- Description of the key steps in the workflow sequence
- Analysis of prompting patterns and strategies used
- Description of UI elements and their functions
- Logical sequence and dependencies between steps
- Important techniques or best practices demonstrated

For the FLOWCHART:
- Create a Mermaid flowchart diagram using the flowchart syntax
- Use clear, concise node labels
- Include all major steps in the process
- Show the relationships and flow between steps
- Include decision points where the workflow branches
- Group related steps if appropriate
- Keep the diagram clean and readable

First provide the summary, then create a section titled "Workflow Diagram" containing only the Mermaid code block for the flowchart.
Use this format for the Mermaid diagram:
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
    %% Add all needed nodes and connections
```

VIDEO ANALYSIS:
{chunk}
"""
