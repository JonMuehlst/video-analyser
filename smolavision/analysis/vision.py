import logging
from typing import Dict, Any, List, Optional

from smolavision.models.base import ModelInterface
from smolavision.batch.types import Batch
from smolavision.analysis.types import AnalysisResult, AnalysisRequest
from smolavision.exceptions import AnalysisError

logger = logging.getLogger(__name__)

def create_analysis_prompt(request: AnalysisRequest) -> str:
    """
    Create a prompt for batch analysis based on the mission type.
    
    Args:
        request: Analysis request containing batch information and mission
        
    Returns:
        Formatted prompt for the vision model
    """
    mission = request.mission.lower()
    language = request.language
    previous_context = request.previous_context
    
    # Base prompt
    prompt = f"Analyze these frames from a video. "
    
    # Add OCR text if available
    ocr_text = ""
    if request.ocr_text and any(request.ocr_text):
        ocr_text = "OCR text extracted from the frames:\n"
        for i, text in enumerate(request.ocr_text):
            if text and text.strip():
                timestamp = request.timestamps[i] if i < len(request.timestamps) else 0
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                ocr_text += f"[{minutes:02d}:{seconds:02d}] {text.strip()}\n"
    
    # Add previous context if available
    context_text = ""
    if previous_context:
        context_text = f"\nContext from previous frames: {previous_context}\n"
    
    # Mission-specific instructions
    if mission == "workflow":
        prompt += f"""
Analyze this sequence of frames to identify workflow steps and processes.
Focus on:
1. Actions being performed
2. Tools or equipment being used
3. Sequence of operations
4. Transitions between steps
5. Any text or instructions visible in the frames

{ocr_text}
{context_text}

Provide a detailed description of the workflow in {language}.
"""
    else:  # Default to general analysis
        prompt += f"""
Provide a detailed description of what you see in these frames.
Focus on:
1. People, objects, and environments
2. Actions and events
3. Text visible in the frames
4. Changes between frames
5. Overall context and situation

{ocr_text}
{context_text}

Provide a comprehensive analysis in {language}.
"""
    
    return prompt

def analyze_batch(
    batch: Batch,
    previous_context: str = "",
    language: str = "English",
    mission: str = "general",
    model: ModelInterface = None,
    batch_id: int = 0
) -> AnalysisResult:
    """
    Analyze a batch of frames using a vision model.
    
    Args:
        batch: Batch of frames to analyze
        previous_context: Context from previous batch analysis
        language: Language for analysis output
        mission: Analysis mission type (general or workflow)
        model: Vision model to use for analysis
        batch_id: Unique identifier for the batch
        
    Returns:
        Analysis result
        
    Raises:
        AnalysisError: If analysis fails
    """
    if not model:
        raise AnalysisError("No vision model provided for analysis")
    
    try:
        # Create analysis request
        request = AnalysisRequest(
            batch_id=batch_id,
            frames=batch.frames,
            timestamps=batch.timestamps,
            image_data=batch.image_data,
            ocr_text=batch.ocr_text if hasattr(batch, "ocr_text") else [],
            previous_context=previous_context,
            language=language,
            mission=mission
        )
        
        # Create prompt
        prompt = create_analysis_prompt(request)
        
        # Analyze images
        analysis_text = model.analyze_images(
            images=batch.image_data,
            prompt=prompt,
            max_tokens=4096
        )
        
        # Extract context for next batch
        context = analysis_text[-1000:] if len(analysis_text) > 1000 else analysis_text
        
        # Create result
        result = AnalysisResult(
            batch_id=batch_id,
            frames=batch.frames,
            timestamps=batch.timestamps,
            analysis_text=analysis_text,
            context=context,
            metadata={
                "mission": mission,
                "language": language
            }
        )
        
        return result
        
    except Exception as e:
        raise AnalysisError(f"Batch analysis failed: {str(e)}") from e
