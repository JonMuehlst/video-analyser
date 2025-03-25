#!/usr/bin/env python
"""
Check dependencies for SmolaVision
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SmolaVision-Setup")

def check_python_packages() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed"""
    required_packages = [
        "numpy",
        "opencv-python",
        "pillow",
        "requests",
        "smolagents",
        "pytesseract"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        # Check for ollama in PATH
        if sys.platform == "win32":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        
        return result.returncode == 0
    except Exception:
        return False

def check_ollama_running() -> bool:
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def list_ollama_models() -> List[str]:
    """List available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        return []
    except Exception:
        return []

def check_tesseract_installed() -> bool:
    """Check if Tesseract OCR is installed"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def main():
    """Check all dependencies for SmolaVision"""
    print("Checking dependencies for SmolaVision...\n")
    
    # Check Python packages
    packages_ok, missing_packages = check_python_packages()
    if packages_ok:
        print("✅ All required Python packages are installed")
    else:
        print("❌ Missing Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Check Ollama
    ollama_installed = check_ollama_installed()
    if ollama_installed:
        print("✅ Ollama is installed")
    else:
        print("❌ Ollama is not installed")
        print("   Install from https://ollama.com")
    
    # Check if Ollama is running
    if ollama_installed:
        ollama_running = check_ollama_running()
        if ollama_running:
            print("✅ Ollama server is running")
            
            # List available models
            models = list_ollama_models()
            if models:
                print(f"   Found {len(models)} models: {', '.join(models[:5])}" + 
                      (f" and {len(models)-5} more..." if len(models) > 5 else ""))
                
                # Check for recommended models
                recommended_models = ["llava", "phi3", "llama3"]
                missing_recommended = [model for model in recommended_models if not any(model in m for m in models)]
                
                if missing_recommended:
                    print("\n   Recommended models not found:")
                    for model in missing_recommended:
                        print(f"   - {model}")
                    print("\n   Pull recommended models with:")
                    for model in missing_recommended:
                        print(f"   ollama pull {model}")
            else:
                print("   No models found. Pull models with:")
                print("   ollama pull llava")
                print("   ollama pull phi3")
        else:
            print("❌ Ollama server is not running")
            print("   Start with: ollama serve")
    
    # Check Tesseract OCR (for OCR functionality)
    tesseract_installed = check_tesseract_installed()
    if tesseract_installed:
        print("✅ Tesseract OCR is installed (required for OCR functionality)")
    else:
        print("❌ Tesseract OCR is not installed (required for OCR functionality)")
        if sys.platform == "win32":
            print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif sys.platform == "darwin":
            print("   Install with: brew install tesseract")
        else:
            print("   Install with: sudo apt install tesseract-ocr")
    
    # Summary
    all_ok = packages_ok and ollama_installed and ollama_running and tesseract_installed
    
    print("\nSummary:")
    if all_ok:
        print("✅ All dependencies are installed and configured correctly")
        print("\nRun SmolaVision with:")
        print("   python run_local.py your_video.mp4")
    else:
        print("❌ Some dependencies are missing or not configured correctly")
        print("   Please install the missing dependencies and try again")

if __name__ == "__main__":
    main()
