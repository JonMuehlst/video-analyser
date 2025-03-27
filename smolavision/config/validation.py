import os
import logging
from typing import Dict, Any, List, Tuple

from smolavision.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

def validate_video_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate video configuration.
    
    Args:
        config: Video configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check frame interval
    frame_interval = config.get("frame_interval")
    if frame_interval is not None and (not isinstance(frame_interval, int) or frame_interval <= 0):
        errors.append(f"Invalid frame interval: {frame_interval}. Must be a positive integer.")
    
    # Check scene threshold
    scene_threshold = config.get("scene_threshold")
    if scene_threshold is not None and (not isinstance(scene_threshold, (int, float)) or scene_threshold <= 0):
        errors.append(f"Invalid scene threshold: {scene_threshold}. Must be a positive number.")
    
    # Check start and end times
    start_time = config.get("start_time", 0)
    end_time = config.get("end_time", 0)
    if start_time < 0:
        errors.append(f"Invalid start time: {start_time}. Must be non-negative.")
    if end_time < 0:
        errors.append(f"Invalid end time: {end_time}. Must be non-negative.")
    if end_time > 0 and start_time >= end_time:
        errors.append(f"Invalid time range: start time ({start_time}) must be less than end time ({end_time}).")
    
    # Check resize width
    resize_width = config.get("resize_width")
    if resize_width is not None and (not isinstance(resize_width, int) or resize_width <= 0):
        errors.append(f"Invalid resize width: {resize_width}. Must be a positive integer.")
    
    return errors

def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check model type
    model_type = config.get("model_type")
    if model_type not in ["anthropic", "openai", "huggingface", "ollama"]:
        errors.append(f"Invalid model type: {model_type}. Must be one of: anthropic, openai, huggingface, ollama.")
    
    # Check API key for cloud models
    if model_type in ["anthropic", "openai"] and not config.get("api_key"):
        errors.append(f"API key is required for {model_type} models.")
    
    # Check Ollama configuration
    if model_type == "ollama":
        ollama_config = config.get("ollama", {})
        if not ollama_config.get("enabled", False):
            errors.append("Ollama is selected as model type but not enabled in configuration.")
        
        # Check if Ollama is installed and running
        if ollama_config.get("enabled", False):
            try:
                import httpx
                base_url = ollama_config.get("base_url", "http://localhost:11434")
                response = httpx.get(f"{base_url}/api/tags", timeout=2)
                if response.status_code != 200:
                    errors.append(f"Ollama server is not responding at {base_url}.")
            except ImportError:
                errors.append("httpx package is required for Ollama integration.")
            except Exception as e:
                errors.append(f"Failed to connect to Ollama server: {str(e)}")
    
    return errors

def validate_analysis_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate analysis configuration.
    
    Args:
        config: Analysis configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check mission
    mission = config.get("mission")
    if mission not in ["general", "workflow"]:
        errors.append(f"Invalid mission: {mission}. Must be one of: general, workflow.")
    
    # Check batch size
    max_batch_size_mb = config.get("max_batch_size_mb")
    if max_batch_size_mb is not None and (not isinstance(max_batch_size_mb, (int, float)) or max_batch_size_mb <= 0):
        errors.append(f"Invalid max batch size: {max_batch_size_mb}. Must be a positive number.")
    
    # Check max images per batch
    max_images_per_batch = config.get("max_images_per_batch")
    if max_images_per_batch is not None and (not isinstance(max_images_per_batch, int) or max_images_per_batch <= 0):
        errors.append(f"Invalid max images per batch: {max_images_per_batch}. Must be a positive integer.")
    
    # Check batch overlap
    batch_overlap_frames = config.get("batch_overlap_frames")
    if batch_overlap_frames is not None and (not isinstance(batch_overlap_frames, int) or batch_overlap_frames < 0):
        errors.append(f"Invalid batch overlap frames: {batch_overlap_frames}. Must be a non-negative integer.")
    
    return errors

def validate_pipeline_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate pipeline configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check pipeline type
    pipeline_type = config.get("pipeline_type")
    if pipeline_type not in ["standard", "segmented"]:
        errors.append(f"Invalid pipeline type: {pipeline_type}. Must be one of: standard, segmented.")
    
    # Check segment length for segmented pipeline
    if pipeline_type == "segmented":
        segment_length = config.get("segment_length")
        if segment_length is not None and (not isinstance(segment_length, int) or segment_length <= 0):
            errors.append(f"Invalid segment length: {segment_length}. Must be a positive integer.")
    
    return errors

def validate_output_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate output configuration.
    
    Args:
        config: Output configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check output formats
    formats = config.get("formats", [])
    valid_formats = ["text", "json", "html", "markdown"]
    
    for format_type in formats:
        if format_type not in valid_formats:
            errors.append(f"Invalid output format: {format_type}. Must be one of: {', '.join(valid_formats)}.")
    
    return errors

def validate_output_dir(output_dir: str) -> List[str]:
    """
    Validate output directory.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check if output directory exists or can be created
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create output directory {output_dir}: {str(e)}")
    elif not os.path.isdir(output_dir):
        errors.append(f"Output path {output_dir} exists but is not a directory.")
    elif not os.access(output_dir, os.W_OK):
        errors.append(f"Output directory {output_dir} is not writable.")
    
    return errors

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the complete configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
        
    Raises:
        ConfigurationError: If validation fails and raise_error is True
    """
    errors = []
    
    # Validate video configuration
    video_config = config.get("video", {})
    errors.extend(validate_video_config(video_config))
    
    # Validate model configuration
    model_config = config.get("model", {})
    errors.extend(validate_model_config(model_config))
    
    # Validate analysis configuration
    analysis_config = config.get("analysis", {})
    errors.extend(validate_analysis_config(analysis_config))
    
    # Validate pipeline configuration
    pipeline_config = config.get("pipeline", {})
    errors.extend(validate_pipeline_config(pipeline_config))
    
    # Validate output configuration
    output_config = config.get("output", {})
    errors.extend(validate_output_config(output_config))
    
    # Validate output directory
    output_dir = config.get("output_dir", "output")
    errors.extend(validate_output_dir(output_dir))
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        for error in errors:
            logger.error(f"Configuration error: {error}")
    
    return is_valid, errors
