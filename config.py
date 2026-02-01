"""Configuration settings for AMEK AI Bot"""

import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Model Configuration
MODELS = {
    "primary": "microsoft/DialoGPT-medium",
    "fallback": "microsoft/DialoGPT-small"
}

# API Configuration
MAX_TOKENS_PER_REQUEST = 1000
REQUEST_TIMEOUT = 30

# Database Configuration
DATABASE_PATH = "chat_history.db"

# Authentication
ADMIN_USERS = ["admin@example.com"]