import os
import sys
import subprocess
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Define required packages with optional import names
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "requests": "requests",
    "pytesseract": "pytesseract",
    "anthropic": "anthropic",
    "openai": "openai",
    "litellm": "litellm",
    "httpx": "httpx",
    "pydantic": "pydantic",
    "transformers": "transformers",
    "ollama": "ollama",
    "python-dotenv": "dotenv",
    "tqdm": "tqdm",
    "scikit-image": "skimage",
}

OPTIONAL_PACKAGES = {
    "pytesseract": "pytesseract", # Required for OCR
}

def check_python_packages() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed"""
    missing_packages = []
    
    for package, import_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        # Check for ollama in PATH
        if sys.platform == "win32":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True, check=False)
        else:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True, check=False)
        
        return result.returncode == 0
    except Exception:
        return False

def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except ImportError:
        logger.warning("requests package not found, cannot check Ollama status.")
        return False
    except Exception:
        return False

def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [model.get("name") for model in models_data if model.get("name")]
        return []
    except ImportError:
        logger.warning("requests package not found, cannot list Ollama models.")
        return []
    except Exception:
        return []

def check_tesseract_installed() -> bool:
    """Check if Tesseract OCR is installed"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except ImportError:
        logger.warning("pytesseract package not found, cannot check Tesseract installation.")
        return False
    except Exception:
        return False

def check_all_dependencies(ollama_base_url: str = "http://localhost:11434") -> Dict[str, str]:
    """
    Check all dependencies for SmolaVision.

    Args:
        ollama_base_url: The base URL for the Ollama server.

    Returns:
        A dictionary containing missing dependencies or configuration issues.
        Keys are dependency names, values are error messages or installation instructions.
    """
    issues = {}

    # Check Python packages
    packages_ok, missing_packages = check_python_packages()
    if not packages_ok:
        issues["Python Packages"] = f"Missing: {', '.join(missing_packages)}. Install with: pip install {' '.join(missing_packages)}"

    # Check Tesseract OCR (optional but needed for OCR)
    if "pytesseract" not in missing_packages:
        if not check_tesseract_installed():
            install_cmd = ""
            if sys.platform == "win32":
                install_cmd = "Install from: https://github.com/UB-Mannheim/tesseract/wiki"
            elif sys.platform == "darwin":
                install_cmd = "Install with: brew install tesseract"
            else:
                install_cmd = "Install with: sudo apt install tesseract-ocr"
            issues["Tesseract OCR"] = f"Not found or not configured correctly. {install_cmd}"

    # Check Ollama
    if "ollama" not in missing_packages:
        if not check_ollama_installed():
            issues["Ollama Installation"] = "Ollama is not installed. Install from https://ollama.com"
        else:
            if not check_ollama_running(ollama_base_url):
                issues["Ollama Server"] = f"Ollama server is not running or not reachable at {ollama_base_url}. Start with: ollama serve"
            else:
                # Check for recommended models if Ollama is running
                models = list_ollama_models(ollama_base_url)
                recommended_models = ["llava", "phi3", "llama3"] # Basic recommendations
                missing_recommended = [model for model in recommended_models if not any(model in m for m in models)]
                if missing_recommended:
                    issues["Ollama Models"] = f"Recommended models not found: {', '.join(missing_recommended)}. Pull with: ollama pull <model_name>"

    return issues
