#!/bin/bash

# Secure AI Bot Startup Script

echo "🔒 Starting Secure AI Bot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and configure your settings."
    exit 1
fi

# Check if secrets.toml exists
if [ ! -f .streamlit/secrets.toml ]; then
    echo "⚠️  Streamlit secrets not found. Please copy .streamlit/secrets.toml.example to .streamlit/secrets.toml"
    exit 1
fi

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Run the secure app
echo "🚀 Launching secure application..."
streamlit run secure_app.py --server.port 8501 --server.address localhost