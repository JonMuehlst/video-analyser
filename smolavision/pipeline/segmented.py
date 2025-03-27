import os
import logging
from typing import Dict, Any, List
from datetime import datetime

from smolavision.pipeline.base import Pipeline
from smolavision.tools.factory import ToolFactory
from smolavision.models.factory import ModelFactory
from smolavision.video.types import Frame
from smolavision.batch.types import Batch
from smolavision.analysis.vision import analyze_batch
from smolavision.analysis.summarization import generate_summary
from smolavision.analysis.types import AnalysisResult
from smolavision.exceptions import PipelineError

logger = logging.getLogger(__name__)

class SegmentedPipeline(Pipeline):
    """Pipeline for processing video in segments."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the segmented pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.video_config = config.get("video", {})
        self.model_config = config.get("model", {})
        self.analysis_config = config.get("analysis", {})
        
        # Initialize models
        self.vision_model = ModelFactory.create_vision_model(self.model_config)
        self.summary_model = ModelFactory.create_summary_model(self.model_config)
        
        # Segment configuration
        self.segment_length = config.get("segment_length", 300)  # 5 minutes by default
        
    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Run the segmented pipeline on a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            PipelineError: If pipeline execution fails
        """
        try:
            # Create output directory
            now = datetime.now()
            formatted_time = now.strftime("%Y%m%d%H%M")
            output_dir = os.path.join(self.config.get("output_dir", "output"), formatted_time)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get video duration
            import cv2
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            video.release()
            
            # Calculate segments
            segments = []
            start_time = self.video_config.get("start_time", 0.0)
            end_time = self.video_config.get("end_time", 0.0)
            if end_time <= 0 or end_time > duration:
                end_time = duration
                
            current_time = start_time
            while current_time < end_time:
                segment_end = min(current_time + self.segment_length, end_time)
                segments.append((current_time, segment_end))
                current_time = segment_end
            
            logger.info(f"Processing video in {len(segments)} segments")
            
            # Process each segment
            all_analyses = []
            
            for i, (segment_start, segment_end) in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}: {segment_start:.1f}s - {segment_end:.1f}s")
                
                # Create tools
                frame_extraction_tool = ToolFactory.create_tool(self.config, "frame_extraction")
                ocr_tool = None
                if self.video_config.get("enable_ocr", False):
                    ocr_tool = ToolFactory.create_tool(self.config, "ocr_extraction")
                batch_tool = ToolFactory.create_tool(self.config, "batch_creation")
                
                # Extract frames for this segment
                segment_config = dict(self.video_config)
                segment_config["start_time"] = segment_start
                segment_config["end_time"] = segment_end
                
                # Update config for this segment
                segment_full_config = dict(self.config)
                segment_full_config["video"] = segment_config
                
                # 1. Extract frames
                logger.info(f"Extracting frames from segment {i+1}")
                frames_str = frame_extraction_tool.use(video_path)
                frames = eval(frames_str)  # Convert string representation back to list
                
                # Skip if no frames extracted
                if not frames:
                    logger.warning(f"No frames extracted for segment {i+1}")
                    continue
                
                # 2. Extract text with OCR if enabled
                if self.video_config.get("enable_ocr", False) and ocr_tool:
                    logger.info(f"Extracting text with OCR for segment {i+1}")
                    frames_str = ocr_tool.use(frames)
                    frames = eval(frames_str)
                
                # 3. Create batches
                logger.info(f"Creating batches for segment {i+1}")
                batches_str = batch_tool.use(frames)
                batches = eval(batches_str)
                
                # 4. Analyze batches
                logger.info(f"Analyzing {len(batches)} batches for segment {i+1}")
                segment_analyses = []
                previous_context = ""
                
                for j, batch_dict in enumerate(batches):
                    logger.info(f"Analyzing batch {j+1}/{len(batches)} of segment {i+1}")
                    
                    # Convert dict to Batch object
                    batch = Batch(**batch_dict)
                    
                    # Analyze batch
                    result = analyze_batch(
                        batch=batch,
                        previous_context=previous_context,
                        language=self.video_config.get("language", "English"),
                        mission=self.analysis_config.get("mission", "general"),
                        model=self.vision_model,
                        batch_id=j
                    )
                    
                    segment_analyses.append(result.analysis_text)
                    previous_context = result.context
                
                # Add segment analyses to all analyses
                all_analyses.extend(segment_analyses)
            
            # 5. Generate summary
            logger.info("Generating summary")
            summary_result = generate_summary(
                analyses=all_analyses,
                language=self.video_config.get("language", "English"),
                mission=self.analysis_config.get("mission", "general"),
                generate_flowchart=self.analysis_config.get("generate_flowchart", False),
                model=self.summary_model,
                output_dir=output_dir
            )
            
            # Return results
            return {
                "summary_text": summary_result["summary_text"],
                "analyses": all_analyses,
                "summary_path": summary_result["summary_path"],
                "full_analysis_path": summary_result["full_analysis_path"],
                "flowchart_path": summary_result.get("flowchart_path"),
                "output_dir": output_dir
            }
            
        except Exception as e:
            raise PipelineError(f"Pipeline execution failed: {str(e)}")
