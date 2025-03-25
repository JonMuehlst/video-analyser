#!/bin/bash
# Simple script to run SmolaVision locally with Ollama

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.com"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    # Wait for Ollama to start
    sleep 5
fi

# Run the application
python run_local.py "$@"
