#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/output/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Check if a video path was provided
if [ $# -eq 0 ]; then
    echo "No video path provided."
    echo "Usage: $0 /path/to/video.mp4 [options]"
    exit 1
fi

VIDEO_PATH="$1"
shift  # Remove the first argument (video path)

# Run the analysis
python "$SCRIPT_DIR/run_local.py" "$VIDEO_PATH" \
    --enable-ocr \
    --generate-flowchart \
    "$@"  # Pass any remaining arguments

echo "Analysis complete. Check the output directory for results."
