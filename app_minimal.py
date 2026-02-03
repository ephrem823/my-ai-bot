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
import requests
from urllib.parse import urlencode, parse_qs
import jwt
from datetime import datetime, timedelta

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
# GOOGLE OAUTH AUTHENTICATION
# ============================================================================

class GoogleOAuth:
    """Handle Google OAuth authentication"""
    
    def __init__(self):
        self.client_id = config.GOOGLE_CLIENT_ID
        self.client_secret = config.GOOGLE_CLIENT_SECRET
        self.redirect_uri = config.GOOGLE_REDIRECT_URI
        self.auth_url = "https://accounts.google.com/o/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def get_auth_url(self) -> str:
        """Generate Google OAuth authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent"
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token"""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        
        response = requests.post(self.token_url, data=data)
        return response.json()
    
    def get_user_info(self, access_token: str) -> dict:
        """Get user information from Google"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(self.userinfo_url, headers=headers)
        return response.json()

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
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        return st.session_state.get("authenticated", False)
    
    @staticmethod
    def get_user_info() -> dict:
        """Get current user information"""
        return st.session_state.get("user_info", {})
    
    @staticmethod
    def logout():
        """Logout user"""
        for key in ["authenticated", "user_info", "access_token"]:
            if key in st.session_state:
                del st.session_state[key]

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
google_oauth = GoogleOAuth()

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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_info" not in st.session_state:
    st.session_state.user_info = {}

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
# AUTHENTICATION FUNCTIONS
# ============================================================================

def handle_oauth_callback():
    """Handle OAuth callback from Google"""
    query_params = st.query_params
    
    if "code" in query_params:
        try:
            # Exchange code for token
            token_data = google_oauth.exchange_code_for_token(query_params["code"])
            
            if "access_token" in token_data:
                # Get user info
                user_info = google_oauth.get_user_info(token_data["access_token"])
                
                # Store in session
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.session_state.access_token = token_data["access_token"]
                
                # Clear URL parameters
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Authentication failed. Please try again.")
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
    
    elif "error" in query_params:
        st.error(f"Authentication cancelled: {query_params.get('error', 'Unknown error')}")

def show_login_page():
    """Display login page with Google OAuth"""
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🪄 AMEK AI</h1>
        <h3>Professional Code Generator</h3>
        <p style="color: #888; margin-bottom: 40px;">Please sign in with your Google account to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        auth_url = google_oauth.get_auth_url()
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="{auth_url}" target="_self">
                <button style="
                    background-color: #4285f4;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 16px;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    text-decoration: none;
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24">
                        <path fill="white" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="white" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="white" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path fill="white" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                    Sign in with Google
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><p style='text-align: center; color: #666; font-size: 14px;'>Secure authentication powered by Google OAuth 2.0</p>", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Check for required environment variables
    if not config.HF_TOKEN:
        st.error("⚠️ HF_TOKEN not found in .env file. Please add your Hugging Face API token.")
        st.stop()
    
    # Handle OAuth callback
    handle_oauth_callback()
    
    # Check authentication
    if not security.is_authenticated():
        show_login_page()
        return
    
    # Get user info
    user_info = security.get_user_info()
    
    # Header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🪄 AMEK AI - Professional Code Generator")
        st.markdown("*Your intelligent coding companion*")
    with col2:
        st.markdown(f"**Welcome, {user_info.get('name', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            security.logout()
            st.rerun()
    
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