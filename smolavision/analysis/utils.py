import logging
from typing import Dict, Any, List, Optional

from smolavision.analysis.types import AnalysisResult

logger = logging.getLogger(__name__)

def extract_context(analysis_text: str, max_length: int = 1000) -> str:
    """
    Extract context from analysis text for use in subsequent analyses.
    
    Args:
        analysis_text: Analysis text to extract context from
        max_length: Maximum length of context
        
    Returns:
        Extracted context
    """
    if not analysis_text:
        return ""
    
    # For now, simply take the last portion of the text
    return analysis_text[-max_length:] if len(analysis_text) > max_length else analysis_text

def merge_analyses(results: List[AnalysisResult]) -> str:
    """
    Merge multiple analysis results into a single text.
    
    Args:
        results: List of analysis results
        
    Returns:
        Merged analysis text
    """
    if not results:
        return ""
    
    # Sort results by batch_id
    sorted_results = sorted(results, key=lambda r: r.batch_id)
    
    # Merge texts
    merged_text = ""
    for result in sorted_results:
        # Add timestamp range for this batch
        if result.timestamps:
            start_time = min(result.timestamps)
            end_time = max(result.timestamps)
            start_min, start_sec = divmod(int(start_time), 60)
            end_min, end_sec = divmod(int(end_time), 60)
            timestamp_range = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
            merged_text += f"\n\n{timestamp_range}\n"
        
        # Add analysis text
        merged_text += result.analysis_text
    
    return merged_text.strip()

def chunk_text(text: str, max_chunk_size: int = 12000) -> List[str]:
    """
    Split text into chunks of maximum size.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, start a new chunk
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
