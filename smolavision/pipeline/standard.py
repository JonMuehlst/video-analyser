import os
import logging
from typing import Dict, Any, List
from datetime import datetime

from smolavision.pipeline.base import Pipeline
from smolavision.tools.factory import ToolFactory
from smolavision.exceptions import PipelineError

logger = logging.getLogger(__name__)

class StandardPipeline(Pipeline):
    """Standard video analysis pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the standard pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.video_config = config.get("video", {})
        self.model_config = config.get("model", {})
        self.analysis_config = config.get("analysis", {})
        
    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
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
            
            # Create tools
            frame_extraction_tool = ToolFactory.create_tool(self.config, "frame_extraction")
            ocr_tool = None
            if self.video_config.get("enable_ocr", False):
                ocr_tool = ToolFactory.create_tool(self.config, "ocr_extraction")
            batch_tool = ToolFactory.create_tool(self.config, "batch_creation")
            vision_tool = ToolFactory.create_tool(self.config, "vision_analysis")
            summary_tool = ToolFactory.create_tool(self.config, "summarization")
            
            # 1. Extract frames
            logger.info(f"Extracting frames from {video_path}")
            frames_str = frame_extraction_tool.use(video_path)
            frames = eval(frames_str)  # Convert string representation back to list
            
            # 2. Extract text with OCR if enabled
            if self.video_config.get("enable_ocr", False) and ocr_tool:
                logger.info("Extracting text with OCR")
                frames_str = ocr_tool.use(frames)
                frames = eval(frames_str)
            
            # 3. Create batches
            logger.info("Creating batches")
            batches_str = batch_tool.use(frames)
            batches = eval(batches_str)
            
            # 4. Analyze batches
            logger.info(f"Analyzing {len(batches)} batches")
            analyses = []
            previous_context = ""
            
            for i, batch in enumerate(batches):
                logger.info(f"Analyzing batch {i+1}/{len(batches)}")
                prompt = f"Analyze these frames from a video. {self.analysis_config.get('mission', 'general')} analysis."
                analysis = vision_tool.use([batch], prompt, previous_context)
                analyses.append(analysis)
                previous_context = analysis[-1000:] if len(analysis) > 1000 else analysis
            
            # 5. Generate summary
            logger.info("Generating summary")
            summary = summary_tool.use(analyses, self.video_config.get("language", "English"))
            
            # Return results
            return {
                "summary_text": summary,
                "analyses": analyses,
                "coherent_summary": os.path.join(output_dir, "video_summary.txt"),
                "full_analysis": os.path.join(output_dir, "video_analysis_full.txt")
            }
            
        except Exception as e:
            raise PipelineError(f"Pipeline execution failed: {str(e)}")
