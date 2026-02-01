#!/bin/bash

# AMEK AI Startup Script

echo "🪄 Starting AMEK AI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Please copy .env.example to .env and configure it."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Create necessary directories
mkdir -p chat_histories

# Run the application
echo "🚀 Launching AMEK AI..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0