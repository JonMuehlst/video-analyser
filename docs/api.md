# SmolaVision API Reference

This document provides a reference for the key APIs in SmolaVision.

## Configuration

### `smolavision.config.loader`

#### `load_config(config_path=None, args=None)`

Load configuration from multiple sources.

**Parameters:**
- `config_path` (str, optional): Path to the configuration file
- `args` (Namespace, optional): Command line arguments

**Returns:**
- Dict[str, Any]: Merged configuration dictionary

#### `load_config_from_file(config_path)`

Load configuration from a JSON file.

**Parameters:**
- `config_path` (str): Path to the configuration file

**Returns:**
- Dict[str, Any]: Configuration dictionary

#### `create_default_config()`

Create a default configuration.

**Returns:**
- Dict[str, Any]: Default configuration dictionary

### `smolavision.config.validation`

#### `validate_config(config)`

Validate the complete configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- Tuple[bool, List[str]]: Tuple of (is_valid, error_messages)

## Video Processing

### `smolavision.video.extractor`

#### `extract_frames(video_path, interval_seconds=10, detect_scenes=True, scene_threshold=30.0, resize_width=None, start_time=0.0, end_time=0.0)`

Extract frames from a video file at regular intervals and detect scene changes.

**Parameters:**
- `video_path` (str): Path to the video file
- `interval_seconds` (int): Extract a frame every N seconds
- `detect_scenes` (bool): Whether to detect scene changes
- `scene_threshold` (float): Threshold for scene change detection
- `resize_width` (int, optional): Width to resize frames to (keeps aspect ratio)
- `start_time` (float): Start time in seconds (0 for beginning)
- `end_time` (float): End time in seconds (0 for entire video)

**Returns:**
- List[Frame]: List of extracted frames

### `smolavision.video.scene_detection`

#### `detect_scene_changes(frames, threshold=30.0)`

Detect scene changes in a list of frames.

**Parameters:**
- `frames` (List[Frame]): List of Frame objects
- `threshold` (float): Threshold for scene change detection

**Returns:**
- List[Frame]: List of Frame objects with scene_change attribute set

## OCR Processing

### `smolavision.ocr.extractor`

#### `extract_text(frames, language="English")`

Extract text from a list of frames using OCR.

**Parameters:**
- `frames` (List[Frame]): List of Frame objects
- `language` (str): Language of the text in the frames

**Returns:**
- List[Frame]: List of Frame objects with ocr_text attribute populated

## Batch Creation

### `smolavision.batch.creator`

#### `create_batches(frames, max_batch_size_mb=10.0, max_images_per_batch=15, overlap_frames=2)`

Create batches of frames for efficient analysis.

**Parameters:**
- `frames` (List[Frame]): List of Frame objects
- `max_batch_size_mb` (float): Maximum size of a batch in megabytes
- `max_images_per_batch` (int): Maximum number of images in a batch
- `overlap_frames` (int): Number of overlapping frames between batches

**Returns:**
- List[Batch]: List of Batch objects

## Models

### `smolavision.models.factory`

#### `ModelFactory.create_vision_model(config)`

Create a vision model instance based on configuration.

**Parameters:**
- `config` (Dict[str, Any]): Model configuration dictionary

**Returns:**
- ModelInterface: A model instance for vision tasks

#### `ModelFactory.create_summary_model(config)`

Create a summary model instance based on configuration.

**Parameters:**
- `config` (Dict[str, Any]): Model configuration dictionary

**Returns:**
- ModelInterface: A model instance for summarization

### `smolavision.models.base.ModelInterface`

Abstract base class for all AI models.

#### `generate_text(prompt, **kwargs)`

Generate text from a prompt.

**Parameters:**
- `prompt` (str): Text prompt
- `**kwargs`: Additional model-specific parameters

**Returns:**
- str: Generated text

#### `analyze_images(images, prompt, max_tokens=4096, **kwargs)`

Analyze images with a text prompt.

**Parameters:**
- `images` (List[str]): List of base64-encoded images
- `prompt` (str): Text prompt for analysis
- `max_tokens` (int): Maximum tokens to generate
- `**kwargs`: Additional model-specific parameters

**Returns:**
- str: Generated analysis text

## Analysis

### `smolavision.analysis.vision`

#### `analyze_batch(batch, previous_context="", language="English", mission="general", model=None, batch_id=0)`

Analyze a batch of frames using a vision model.

**Parameters:**
- `batch` (Batch): Batch of frames to analyze
- `previous_context` (str): Context from previous batch analysis
- `language` (str): Language for analysis output
- `mission` (str): Analysis mission type (general or workflow)
- `model` (ModelInterface): Vision model to use for analysis
- `batch_id` (int): Unique identifier for the batch

**Returns:**
- AnalysisResult: Analysis result

### `smolavision.analysis.summarization`

#### `generate_summary(analyses, language="English", mission="general", generate_flowchart=False, model=None, output_dir="output")`

Generate a summary from batch analyses.

**Parameters:**
- `analyses` (List[str]): List of analysis texts
- `language` (str): Language for summary output
- `mission` (str): Summary mission type (general or workflow)
- `generate_flowchart` (bool): Whether to generate a flowchart
- `model` (ModelInterface): Summary model to use
- `output_dir` (str): Directory to save output files

**Returns:**
- Dict[str, Any]: Dictionary with summary results

## Pipeline

### `smolavision.pipeline.factory`

#### `create_pipeline(config)`

Create a pipeline instance based on configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- Pipeline: Pipeline instance

### `smolavision.pipeline.base.Pipeline`

Abstract base class for all pipelines.

#### `run(video_path)`

Run the pipeline on a video.

**Parameters:**
- `video_path` (str): Path to the video file

**Returns:**
- Dict[str, Any]: Dictionary with analysis results

## Tools (Internal Components)

*Note: Tools are primarily used internally by pipelines. Direct usage is less common.*

### `smolavision.tools.factory`

#### `ToolFactory.create_tool(config, tool_name)`

Creates a tool instance for internal pipeline use.

### `smolavision.tools.base.Tool`

Abstract base class for internal tools.

#### `use(*args, **kwargs)`

Executes the tool's specific logic.

## Output

### `smolavision.output.formatter`

#### `format_output(result, format_type=OutputFormat.TEXT)`

Format analysis results according to the specified format.

**Parameters:**
- `result` (OutputResult): Analysis results
- `format_type` (OutputFormat): Output format

**Returns:**
- str: Formatted output string

### `smolavision.output.writer`

#### `write_output(result, output_dir, formats=[OutputFormat.TEXT])`

Write analysis results to files in the specified formats.

**Parameters:**
- `result` (OutputResult): Analysis results
- `output_dir` (str): Directory to write files to
- `formats` (List[OutputFormat]): List of output formats to generate

**Returns:**
- Dict[str, str]: Dictionary mapping format names to file paths

### `smolavision.output.flowchart`

#### `generate_flowchart(text)`

Extract or generate a flowchart from analysis text.

**Parameters:**
- `text` (str): Analysis text that may contain a flowchart

**Returns:**
- str or None: Mermaid flowchart syntax or None if no flowchart could be generated
