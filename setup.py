#!/usr/bin/env python
"""
Setup script for SmolaVision
"""

from setuptools import setup, find_packages

setup(
    name="smolavision",
    version="0.1.0",
    description="A system for analyzing videos using AI vision models",
    author="SmolaVision Team",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.8"],
        "anthropic": ["anthropic>=0.5.0"],
        "openai": ["openai>=1.0.0"],
        "ollama": ["litellm>=1.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.9.0",
        ],
        "all": ["smolavision[ocr,anthropic,openai,ollama,dev]"],
    },
    entry_points={
        "console_scripts": [
            "smolavision=smolavision.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
