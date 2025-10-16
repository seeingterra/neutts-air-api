#!/bin/bash
# Quick start script for NeuTTS Air API Server

set -e

echo "🚀 Starting NeuTTS Air API Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing requirements..."
    pip install -r requirements.txt
    pip install -r requirements-api.txt
    echo "✅ Requirements installed"
fi

# Start the server
echo ""
echo "🎙️ Starting API server on http://127.0.0.1:8011"
echo "🌐 Web GUI will be available at http://127.0.0.1:8011/gui"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python api_server.py --host 127.0.0.1 --port 8011
