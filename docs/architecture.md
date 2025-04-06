# SmolaVision Architecture

This document describes the high-level architecture of SmolaVision, a system for analyzing videos using AI vision models.

## System Overview

SmolaVision is designed to process videos by extracting frames, analyzing them with AI vision models, and generating coherent summaries. The system is modular and extensible, allowing for different AI models, processing pipelines, and output formats.

## Core Components

### 1. Video Processing

The video processing module is responsible for extracting frames from videos and detecting scene changes. It uses OpenCV for video manipulation and provides a flexible interface for controlling frame extraction parameters.

Key components:
- `FrameExtractor`: Interface for video frame extraction
- `DefaultFrameExtractor`: Default implementation of frame extraction
- Scene change detection using histogram comparison

### 2. OCR Processing

The OCR module extracts text from video frames using Tesseract OCR. It supports multiple languages and provides confidence scores for extracted text.

Key components:
- `extract_text`: Function to extract text from frames
- Language mapping utilities for Tesseract

### 3. Batch Creation

The batch creation module groups frames into batches for efficient processing by AI models. It manages batch size constraints based on memory usage and model limitations.

Key components:
- `create_batches`: Function to create batches from frames
- Batch size calculation utilities
- Overlap management between batches

### 4. AI Models

The models module provides a unified interface for different AI vision models, including cloud-based models (Anthropic Claude, OpenAI GPT-4) and local models (Ollama, Hugging Face).

Key components:
- `ModelInterface`: Abstract base class for all models
- Model implementations for different providers
- `ModelFactory`: Factory for creating model instances

### 5. Analysis Pipeline

The pipeline module orchestrates the entire analysis process, from frame extraction to summary generation. It provides different pipeline implementations for various use cases.

Key components:
- `Pipeline`: Abstract base class for all pipelines
- `StandardPipeline`: Standard video analysis pipeline
- `SegmentedPipeline`: Pipeline for processing long videos in segments

### 6. Tools (Internal Component)

*Note: The `tools` module is primarily an internal implementation detail used by the pipelines. Direct interaction is typically done via the main pipeline or CLI.*

The tools module provides a collection of components that perform specific tasks within the analysis pipeline.

Key components:
- `Tool`: Abstract base class for tools
- Implementations for frame extraction, OCR, batch creation, vision analysis, and summarization
- `ToolFactory`: Factory for creating tool instances used by pipelines

### 7. Output Generation

The output module handles formatting and saving analysis results in various formats, including text, JSON, HTML, and Markdown.

Key components:
- Output formatting utilities
- File writing functions
- Flowchart generation for workflow analysis

## Data Flow (Standard Pipeline)

1. Video input is processed by the `FrameExtractionTool` (via `smolavision.video.extractor`) to extract frames.
2. If OCR is enabled, the `OCRExtractionTool` (via `smolavision.ocr.extractor`) extracts text from the frames.
3. The `BatchCreationTool` (via `smolavision.batch.creator`) groups frames into batches.
4. Each batch is analyzed by the `VisionAnalysisTool` (via `smolavision.analysis.vision`) using the selected AI model.
5. The `SummarizationTool` (via `smolavision.analysis.summarization`) generates a coherent summary from the batch analyses.
6. Results are formatted (`smolavision.output.formatter`) and saved (`smolavision.output.writer`) to the output directory.

## Configuration System

SmolaVision uses a flexible configuration system that supports multiple sources:

1. Default configuration
2. Configuration file (JSON)
3. Environment variables
4. Command line arguments

Configuration is validated before use to ensure all parameters are valid.

## Extension Points

SmolaVision is designed to be extensible. Key extension points include:

1. Adding new model implementations by extending `ModelInterface`
2. Creating custom tools by extending `Tool`
3. Implementing new pipelines by extending `Pipeline`
4. Adding new output formats

## Dependencies

SmolaVision relies on the following key dependencies:

- OpenCV for video processing
- Tesseract for OCR
- Anthropic, OpenAI, and Hugging Face libraries for cloud-based models
- Ollama for local model inference
- Pydantic for data validation
- Various Python standard libraries
