#!/bin/bash

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable is not set."
    echo "Please set it with: export ANTHROPIC_API_KEY=your_api_key"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./output/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run the analysis
python run_anthropic.py "/home/jonm/Videos/ai_use_demo.mp4" \
    --output-dir "$OUTPUT_DIR" \
    --language "Hebrew" \
    --frame-interval 10 \
    --scene-threshold 30.0 \
    --vision-model "claude-3-opus-20240229" \
    --summary-model "claude-3-5-sonnet-20240620"

echo "Analysis complete. Results saved to $OUTPUT_DIR"
