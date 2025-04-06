# In summarize-video/run_gemini.py

#!/usr/bin/env python
import os
import dotenv
from smolavision.pipeline import run_smolavision

# Load .env file (contains your API keys)
dotenv.load_dotenv()

# Run analysis
result = run_smolavision(
    video_path="",
    config={
        "model": {
            "model_type": "gemini",
            # API key will be loaded from .env
            # --- Use a valid model name from the list ---
            "vision_model": "gemini-2.0-flash-exp", # "gemini-2.5-pro-exp-03-25", # Changed
            "summary_model": "gemini-2.0-flash-exp", # "gemini-2.5-pro-exp-03-25" # Changed
            # You could also try "gemini-1.5-pro-latest" here
        },
        "video": {
            "frame_interval": 5,
            "detect_scenes": True,
            "scene_threshold": 20.0,
            "enable_ocr": True,
            "language": "English"
        },
        "analysis": {
            "mission": "workflow",
            "generate_flowchart": True
        }
    }
)

# Print results
print("\nAnalysis Summary:")
print(result["summary_text"])
print(f"\nResults saved to: {result['output_dir']}")