@echo off
echo Running SmolaVision with local Ollama models...
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8 or newer.
    goto :end
)

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Ollama server not running! Please start Ollama first.
    goto :end
)

REM Check if video path was provided
if "%~1"=="" (
    echo No video path provided.
    echo Usage: run_local.bat "path\to\your\video.mp4"
    goto :end
)

REM Run the script with the provided video path
python run_local.py "%~1"

:end
echo.
pause
