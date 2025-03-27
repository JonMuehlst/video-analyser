import os
import logging
from typing import Dict, Any, List, Optional

from smolavision.models.base import ModelInterface
from smolavision.analysis.types import AnalysisResult
from smolavision.exceptions import SummarizationError

logger = logging.getLogger(__name__)

def create_summary_prompt(
    analyses: List[str],
    language: str = "English",
    mission: str = "general",
    generate_flowchart: bool = False
) -> str:
    """
    Create a prompt for generating a summary based on the mission type.
    
    Args:
        analyses: List of analysis texts
        language: Language for summary output
        mission: Summary mission type (general or workflow)
        generate_flowchart: Whether to generate a flowchart
        
    Returns:
        Formatted prompt for the summary model
    """
    # Combine all analyses
    combined_text = "\n\n".join(analyses)
    
    # Base prompt
    prompt = f"I'll provide you with analyses of different segments of a video. "
    
    # Mission-specific instructions
    if mission == "workflow":
        prompt += f"""
Your task is to create a coherent summary of the workflow shown in the video.

The analyses describe different parts of the video:

{combined_text}

Based on these analyses, provide:
1. A comprehensive summary of the entire workflow
2. The main steps in the process
3. Tools and equipment used
4. Key observations and insights
"""
        if generate_flowchart:
            prompt += """
5. A flowchart in Mermaid syntax that visualizes the workflow steps. Use the following format:
```mermaid
flowchart TD
    A[Step 1] --> B[Step 2]
    B --> C[Step 3]
    ...
```
"""
    else:  # Default to general analysis
        prompt += f"""
Your task is to create a coherent summary of the video content.

The analyses describe different parts of the video:

{combined_text}

Based on these analyses, provide:
1. A comprehensive summary of the entire video
2. Main subjects, objects, and environments
3. Key events and actions
4. Important text or information visible in the video
5. Overall context and significance
"""
    
    prompt += f"\nProvide your response in {language}."
    
    return prompt

def generate_summary(
    analyses: List[str],
    language: str = "English",
    mission: str = "general",
    generate_flowchart: bool = False,
    model: ModelInterface = None,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Generate a summary from batch analyses.
    
    Args:
        analyses: List of analysis texts
        language: Language for summary output
        mission: Summary mission type (general or workflow)
        generate_flowchart: Whether to generate a flowchart
        model: Summary model to use
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with summary results
        
    Raises:
        SummarizationError: If summarization fails
    """
    if not model:
        raise SummarizationError("No summary model provided")
    
    try:
        # Create prompt
        prompt = create_summary_prompt(
            analyses=analyses,
            language=language,
            mission=mission,
            generate_flowchart=generate_flowchart
        )
        
        # Generate summary
        summary_text = model.generate_text(prompt=prompt)
        
        # Save summary to file
        summary_path = os.path.join(output_dir, "video_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        # Save full analysis to file
        full_analysis_path = os.path.join(output_dir, "video_analysis_full.txt")
        with open(full_analysis_path, "w", encoding="utf-8") as f:
            f.write("\n\n" + "="*80 + "\n\n".join(analyses))
        
        # Extract flowchart if generated
        flowchart_path = None
        if generate_flowchart and "```mermaid" in summary_text:
            flowchart_text = summary_text.split("```mermaid")[1].split("```")[0].strip()
            flowchart_path = os.path.join(output_dir, "workflow_flowchart.mmd")
            with open(flowchart_path, "w", encoding="utf-8") as f:
                f.write(flowchart_text)
        
        return {
            "summary_text": summary_text,
            "summary_path": summary_path,
            "full_analysis_path": full_analysis_path,
            "flowchart_path": flowchart_path if flowchart_path else None
        }
        
    except Exception as e:
        raise SummarizationError(f"Summarization failed: {str(e)}") from e
