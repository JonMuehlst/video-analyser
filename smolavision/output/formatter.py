import json
import logging
from typing import Dict, Any, List, Optional

from smolavision.output.types import OutputFormat, OutputResult

logger = logging.getLogger(__name__)

def format_output(result: OutputResult, format_type: OutputFormat = OutputFormat.TEXT) -> str:
    """
    Format analysis results according to the specified format.
    
    Args:
        result: Analysis results
        format_type: Output format
        
    Returns:
        Formatted output string
    """
    if format_type == OutputFormat.TEXT:
        return _format_as_text(result)
    elif format_type == OutputFormat.JSON:
        return _format_as_json(result)
    elif format_type == OutputFormat.HTML:
        return _format_as_html(result)
    elif format_type == OutputFormat.MARKDOWN:
        return _format_as_markdown(result)
    else:
        logger.warning(f"Unsupported output format: {format_type}. Falling back to TEXT.")
        return _format_as_text(result)

def _format_as_text(result: OutputResult) -> str:
    """Format results as plain text."""
    output = "="*80 + "\n"
    output += "VIDEO ANALYSIS SUMMARY\n"
    output += "="*80 + "\n\n"
    output += result.summary + "\n\n"
    
    output += "="*80 + "\n"
    output += "FULL ANALYSIS\n"
    output += "="*80 + "\n\n"
    output += result.full_analysis + "\n"
    
    if result.flowchart:
        output += "\n" + "="*80 + "\n"
        output += "WORKFLOW FLOWCHART (MERMAID SYNTAX)\n"
        output += "="*80 + "\n\n"
        output += result.flowchart + "\n"
    
    return output

def _format_as_json(result: OutputResult) -> str:
    """Format results as JSON."""
    data = {
        "summary": result.summary,
        "full_analysis": result.full_analysis,
        "flowchart": result.flowchart,
        "metadata": result.metadata,
        "files": result.files
    }
    return json.dumps(data, indent=2)

def _format_as_html(result: OutputResult) -> str:
    """Format results as HTML."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Results</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1, h2 { color: #333; }
        .section { margin-bottom: 30px; }
        pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .mermaid { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
    </style>
"""
    
    if result.flowchart:
        html += """    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true});</script>
"""
    
    html += """</head>
<body>
    <h1>Video Analysis Results</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <div>
"""
    
    # Add summary with paragraphs
    for paragraph in result.summary.split("\n\n"):
        if paragraph.strip():
            html += f"        <p>{paragraph}</p>\n"
    
    html += """        </div>
    </div>
    
    <div class="section">
        <h2>Full Analysis</h2>
        <pre>"""
    
    html += result.full_analysis
    
    html += """</pre>
    </div>
"""
    
    if result.flowchart:
        html += """    <div class="section">
        <h2>Workflow Flowchart</h2>
        <div class="mermaid">
"""
        html += result.flowchart
        html += """
        </div>
    </div>
"""
    
    html += """</body>
</html>"""
    
    return html

def _format_as_markdown(result: OutputResult) -> str:
    """Format results as Markdown."""
    md = "# Video Analysis Results\n\n"
    
    md += "## Summary\n\n"
    md += result.summary + "\n\n"
    
    md += "## Full Analysis\n\n"
    md += "```\n" + result.full_analysis + "\n```\n\n"
    
    if result.flowchart:
        md += "## Workflow Flowchart\n\n"
        md += "```mermaid\n" + result.flowchart + "\n```\n\n"
    
    return md
