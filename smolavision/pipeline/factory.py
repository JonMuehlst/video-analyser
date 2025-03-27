from typing import Dict, Any
from smolavision.pipeline.base import Pipeline
from smolavision.pipeline.standard import StandardPipeline
from smolavision.pipeline.segmented import SegmentedPipeline
from smolavision.exceptions import ConfigurationError

def create_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Create a pipeline instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Pipeline instance
        
    Raises:
        ConfigurationError: If pipeline creation fails
    """
    pipeline_type = config.get("pipeline_type", "standard")
    
    if pipeline_type == "standard":
        return StandardPipeline(config)
    elif pipeline_type == "segmented":
        return SegmentedPipeline(config)
    else:
        raise ConfigurationError(f"Unsupported pipeline type: {pipeline_type}")
