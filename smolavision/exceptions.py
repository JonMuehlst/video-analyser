# smolavision/exceptions.py

class SmolaVisionError(Exception):
    """Base class for all SmolaVision exceptions."""
    pass


class ConfigurationError(SmolaVisionError):
    """Raised when there is an error in the configuration."""
    pass


class VideoProcessingError(SmolaVisionError):
    """Raised when there is an error during video processing."""
    pass


class OCRProcessingError(SmolaVisionError):
    """Raised when there is an error during OCR processing."""
    pass


class ModelError(SmolaVisionError):
    """Raised when there is an error with the AI model."""
    pass


class PipelineError(SmolaVisionError):
    """Raised when there is an error during pipeline execution."""
    pass


class ToolError(SmolaVisionError):
    """Raised when there is an error during tool execution."""
    pass


class AnalysisError(SmolaVisionError):
    """Raised when there is an error during batch analysis."""
    pass


class SummarizationError(SmolaVisionError):
    """Raised when there is an error during summarization."""
    pass


class OutputError(SmolaVisionError):
    """Raised when there is an error during output generation."""
    pass
