import os
import logging
from typing import Dict, Any, List, Optional

from smolavision.output.types import OutputFormat, OutputResult
from smolavision.output.formatter import format_output
from smolavision.exceptions import OutputError

logger = logging.getLogger(__name__)

def write_output(
    result: OutputResult,
    output_dir: str,
    formats: List[OutputFormat] = [OutputFormat.TEXT]
) -> Dict[str, str]:
    """
    Write analysis results to files in the specified formats.
    
    Args:
        result: Analysis results
        output_dir: Directory to write files to
        formats: List of output formats to generate
        
    Returns:
        Dictionary mapping format names to file paths
        
    Raises:
        OutputError: If writing output fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write files in each format
        output_files = {}
        
        for format_type in formats:
            formatted_output = format_output(result, format_type)
            
            # Determine file extension
            if format_type == OutputFormat.TEXT:
                ext = "txt"
            elif format_type == OutputFormat.JSON:
                ext = "json"
            elif format_type == OutputFormat.HTML:
                ext = "html"
            elif format_type == OutputFormat.MARKDOWN:
                ext = "md"
            else:
                ext = "txt"
            
            # Write to file
            file_path = os.path.join(output_dir, f"analysis.{ext}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            
            output_files[format_type.value] = file_path
            logger.info(f"Wrote {format_type.value} output to {file_path}")
        
        # Write summary and full analysis as separate files
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(result.summary)
        output_files["summary"] = summary_path
        
        full_analysis_path = os.path.join(output_dir, "full_analysis.txt")
        with open(full_analysis_path, "w", encoding="utf-8") as f:
            f.write(result.full_analysis)
        output_files["full_analysis"] = full_analysis_path
        
        # Write flowchart if available
        if result.flowchart:
            flowchart_path = os.path.join(output_dir, "flowchart.mmd")
            with open(flowchart_path, "w", encoding="utf-8") as f:
                f.write(result.flowchart)
            output_files["flowchart"] = flowchart_path
        
        return output_files
        
    except Exception as e:
        raise OutputError(f"Failed to write output: {str(e)}") from e
