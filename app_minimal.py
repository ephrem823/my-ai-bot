import streamlit as st
import os
import datetime
import json
import uuid
import hashlib
import secrets
import time
import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import defaultdict

# Third-party imports
from huggingface_hub import InferenceClient
import bleach

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration management"""
    # AI Models
    MODELS = {
        "primary": "deepseek-ai/DeepSeek-V3",
        "fast_check": "zai-org/GLM-4.7-Flash"
    }
    
    # Security
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "efaxalemayehu@gmail.com")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    HF_TOKEN = st.secrets["HF_TOKEN"]
    HF_TOKEN_SECONDARY = st.secrets.get("HF_TOKEN_SECONDARY", "")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
    # Google OAuth
    GOOGLE_CLIENT_ID = st.secrets["auth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["auth"]["client_secret"]
    GOOGLE_REDIRECT_URI = st.secrets["auth"]["redirect_uri"]
    COOKIE_SECRET = st.secrets["auth"]["cookie_secret"]
    SERVER_METADATA_URL = st.secrets["auth"]["server_metadata_url"]
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20"))
    MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2500"))
    
    # Storage
    DATABASE_PATH = os.getenv("DATABASE_PATH", "chats.db")
    CHAT_HISTORY_DIR = "chat_histories"
    
    # Features
    ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    ENABLE_FILE_UPLOAD = os.getenv("ENABLE_FILE_UPLOAD", "true").lower() == "true"
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    # Cost Management
    MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET_USD", "100.0"))
    TOKEN_COSTS = {
        "deepseek-ai/DeepSeek-V3": 0.00002,
        "zai-org/GLM-4.7-Flash": 0.000001
    }

config = Config()

# ============================================================================
# SECURITY & UTILITIES
# ============================================================================

class SecurityManager:
    """Handles security operations"""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Prevent XSS and injection attacks"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = bleach.clean(text, tags=[], strip=True)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r"('\\s*(or|and)\\s*')",
            r"(--)",
            r"(/\\*|\\*/)",
            r"(;\\s*drop\\s+table)",
            r"(;\\s*delete\\s+from)"
        ]
        for pattern in sql_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        return text.strip()

# ============================================================================
# AI CLIENT MANAGER
# ============================================================================

class AIClientManager:
    """Manage AI model connections"""
    
    def __init__(self):
        self.primary_client = InferenceClient(api_key=config.HF_TOKEN)
        self.backup_client = InferenceClient(api_key=config.HF_TOKEN_SECONDARY) if config.HF_TOKEN_SECONDARY else None
        self.use_backup = False
    
    def get_client(self) -> InferenceClient:
        """Get active client"""
        if self.use_backup and self.backup_client:
            return self.backup_client
        return self.primary_client
    
    def switch_to_backup(self):
        """Switch to backup token"""
        if self.backup_client:
            self.use_backup = True

# Initialize
security = SecurityManager()
ai_manager = AIClientManager()

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AMEK AI - Professional Code Generator",
    layout="wide",
    page_icon="🪄",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Base Theme */
    .stApp {
        background-color: #131314;
        color: #E3E3E3;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E1F20 !important;
        border: none;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-radius: 32px !important;
        background-color: #1E1F20 !important;
        border: 1px solid #3C4043 !important;
    }
    
    .stChatMessage {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def generate_ai_response(prompt: str, context: str = "", model: str = None) -> Tuple[str, int, float]:
    """Generate AI response with error handling and metrics"""
    start_time = time.time()
    
    if not model:
        model = config.MODELS["primary"]
    
    try:
        client = ai_manager.get_client()
        
        # Prepare messages
        messages = [
            {"role": "system", "content": """You are AMEK, a professional AI code generator and assistant. 
            You provide high-quality, secure, and well-documented code solutions.
            Always explain your code and include best practices.
            Format code blocks with proper syntax highlighting."""}
        ]
        
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=config.MAX_TOKENS_PER_REQUEST,
            temperature=0.7,
            stream=False
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else len(content.split()) * 1.3
        
        processing_time = time.time() - start_time
        
        return content, int(tokens_used), processing_time
        
    except Exception as e:
        # Try backup client
        if not ai_manager.use_backup and ai_manager.backup_client:
            ai_manager.switch_to_backup()
            return generate_ai_response(prompt, context, model)
        
        processing_time = time.time() - start_time
        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\\n\\nError: {str(e)}"
        return error_msg, 0, processing_time

def display_message(message: dict, is_user: bool = False):
    """Display a chat message with proper formatting"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.markdown(message["content"])
        else:
            # Display AI response with code highlighting
            content = message["content"]
            
            # Check if content contains code blocks
            if "```" in content:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Regular text
                        if part.strip():
                            st.markdown(part)
                    else:  # Code block
                        lines = part.split('\\n')
                        language = lines[0] if lines[0] else "text"
                        code = '\\n'.join(lines[1:]) if len(lines) > 1 else part
                        
                        if code.strip():
                            st.code(code, language=language)
            else:
                st.markdown(content)
            
            # Show metadata
            if message.get("tokens_used") or message.get("processing_time"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if message.get("model_used"):
                        st.caption(f"🤖 {message['model_used']}")
                with col2:
                    if message.get("tokens_used"):
                        st.caption(f"📊 {message['tokens_used']} tokens")
                with col3:
                    if message.get("processing_time"):
                        st.caption(f"⏱️ {message['processing_time']:.1f}s")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Check for required environment variables
    if not config.HF_TOKEN:
        st.error("⚠️ HF_TOKEN not found in .env file. Please add your Hugging Face API token.")
        st.stop()
    
    # Header
    st.title("🪄 AMEK AI - Professional Code Generator")
    st.markdown("*Your intelligent coding companion*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Model selection
        selected_model = st.selectbox(
            "🤖 AI Model",
            options=list(config.MODELS.values()),
            index=0
        )
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        quick_prompts = [
            "Create a Python function",
            "Debug this code",
            "Optimize performance",
            "Add error handling",
            "Write unit tests"
        ]
        
        for prompt in quick_prompts:
            if st.button(prompt, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message, is_user=(message["role"] == "user"))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about coding..."):
        # Sanitize input
        prompt = security.sanitize_input(prompt)
        
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_message(user_message, is_user=True)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response_content, tokens_used, processing_time = generate_ai_response(
                    prompt, 
                    model=selected_model
                )
                
                # Create assistant message with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "tokens_used": tokens_used,
                    "processing_time": processing_time,
                    "model_used": selected_model.split("/")[-1]
                }
                
                st.session_state.messages.append(assistant_message)
                display_message(assistant_message)

if __name__ == "__main__":
    main()