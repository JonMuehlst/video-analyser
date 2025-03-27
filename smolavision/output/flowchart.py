import re
import logging
from typing import Dict, Any, List, Optional

from smolavision.exceptions import OutputError

logger = logging.getLogger(__name__)

def generate_flowchart(text: str) -> Optional[str]:
    """
    Extract or generate a flowchart from analysis text.
    
    Args:
        text: Analysis text that may contain a flowchart
        
    Returns:
        Mermaid flowchart syntax or None if no flowchart could be generated
        
    Raises:
        OutputError: If flowchart generation fails
    """
    try:
        # First, try to extract an existing flowchart
        flowchart = extract_mermaid_flowchart(text)
        if flowchart:
            return flowchart
        
        # If no flowchart found, try to generate one from workflow steps
        steps = extract_workflow_steps(text)
        if steps:
            return generate_mermaid_flowchart(steps)
        
        return None
        
    except Exception as e:
        raise OutputError(f"Failed to generate flowchart: {str(e)}") from e

def extract_mermaid_flowchart(text: str) -> Optional[str]:
    """
    Extract a Mermaid flowchart from text.
    
    Args:
        text: Text that may contain a Mermaid flowchart
        
    Returns:
        Extracted flowchart or None if not found
    """
    # Look for Mermaid flowchart between ```mermaid and ``` markers
    pattern = r"```mermaid\s*(flowchart\s+[^`]+)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return None

def extract_workflow_steps(text: str) -> List[str]:
    """
    Extract workflow steps from analysis text.
    
    Args:
        text: Analysis text
        
    Returns:
        List of workflow steps
    """
    steps = []
    
    # Look for numbered steps (e.g., "1. Step one", "Step 1:", etc.)
    step_patterns = [
        r"\b(\d+)\.\s+([^\n]+)",  # "1. Step description"
        r"Step\s+(\d+):\s+([^\n]+)",  # "Step 1: Description"
        r"(\d+)\)\s+([^\n]+)"  # "1) Step description"
    ]
    
    for pattern in step_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Sort by step number and add descriptions
            sorted_steps = sorted(matches, key=lambda x: int(x[0]))
            steps = [step[1].strip() for step in sorted_steps]
            break
    
    return steps

def generate_mermaid_flowchart(steps: List[str]) -> str:
    """
    Generate a Mermaid flowchart from workflow steps.
    
    Args:
        steps: List of workflow steps
        
    Returns:
        Mermaid flowchart syntax
    """
    if not steps:
        return ""
    
    # Create flowchart header
    flowchart = "flowchart TD\n"
    
    # Add nodes
    for i, step in enumerate(steps):
        node_id = chr(65 + i)  # A, B, C, ...
        flowchart += f"    {node_id}[\"{step}\"]\n"
    
    # Add connections
    for i in range(len(steps) - 1):
        node_id1 = chr(65 + i)
        node_id2 = chr(65 + i + 1)
        flowchart += f"    {node_id1} --> {node_id2}\n"
    
    return flowchart
