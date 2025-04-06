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
        
        # Initialize models
        self.vision_model = ModelFactory.create_vision_model(self.model_config)
        self.summary_model = ModelFactory.create_summary_model(self.model_config)
        
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
            
            # 1. Extract frames
            logger.info(f"Extracting frames from {video_path}")
            # Tool now returns a list of dicts directly
            frames: List[Dict[str, Any]] = frame_extraction_tool.use(video_path)
            
            # 2. Extract text with OCR if enabled
            if self.video_config.get("enable_ocr", False) and ocr_tool:
                logger.info("Extracting text with OCR")
                # Tool now returns a list of dicts directly
                frames = ocr_tool.use(frames)
            
            # 3. Create batches
            logger.info("Creating batches")
            # Tool now returns a list of dicts directly
            batches: List[Dict[str, Any]] = batch_tool.use(frames)
            
            # 4. Analyze batches
            logger.info(f"Analyzing {len(batches)} batches")
            analysis_results = []
            previous_context = ""
            
            for i, batch_dict in enumerate(batches):
                logger.info(f"Analyzing batch {i+1}/{len(batches)}")
                
                # Convert dict to Batch object
                batch = Batch(**batch_dict)
                
                # Analyze batch
                result = analyze_batch(
                    batch=batch,
                    previous_context=previous_context,
                    language=self.video_config.get("language", "English"),
                    mission=self.analysis_config.get("mission", "general"),
                    model=self.vision_model,
                    batch_id=i
                )
                
                analysis_results.append(result)
                previous_context = result.context
            
            # 5. Generate summary
            logger.info("Generating summary")
            analyses = [result.analysis_text for result in analysis_results]
            
            summary_result = generate_summary(
                analyses=analyses,
                language=self.video_config.get("language", "English"),
                mission=self.analysis_config.get("mission", "general"),
                generate_flowchart=self.analysis_config.get("generate_flowchart", False),
                model=self.summary_model,
                output_dir=output_dir
            )
            
            # Return results
            return {
                "summary_text": summary_result["summary_text"],
                "analyses": analyses,
                "summary_path": summary_result["summary_path"],
                "full_analysis_path": summary_result["full_analysis_path"],
                "flowchart_path": summary_result.get("flowchart_path"),
                "output_dir": output_dir
            }
            
        except Exception as e:
            raise PipelineError(f"Pipeline execution failed: {str(e)}")
