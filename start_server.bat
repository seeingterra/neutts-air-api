@echo off
REM Quick start script for NeuTTS Air API Server (Windows)

echo 🚀 Starting NeuTTS Air API Server...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Virtual environment not found. Creating one...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo 📥 Installing requirements...
    pip install -r requirements.txt
    pip install -r requirements-api.txt
    echo ✅ Requirements installed
)

REM Start the server
echo.
echo 🎙️ Starting API server on http://127.0.0.1:8011
echo 🌐 Web GUI will be available at http://127.0.0.1:8011/gui
echo.
echo Press Ctrl+C to stop the server
echo.

python api_server.py --host 127.0.0.1 --port 8011
