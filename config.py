import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_TOKEN_SECONDARY = os.getenv('HF_TOKEN_SECONDARY')

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')

# Available Models - Text Generation Models Only
MODELS = {
    'dialogpt': 'microsoft/DialoGPT-medium',
    'gpt2': 'gpt2',
    'distilgpt2': 'distilgpt2'
}