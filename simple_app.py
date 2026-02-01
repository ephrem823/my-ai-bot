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
import pandas as pd

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
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_TOKEN_SECONDARY = os.getenv("HF_TOKEN_SECONDARY", "")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
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
# SIMPLE AUTHENTICATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

def login_form():
    """Simple login form"""
    st.markdown("### 🔐 Login to AMEK AI")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your.email@example.com")
        name = st.text_input("Name", placeholder="Your Name")
        submitted = st.form_submit_button("Login", type="primary")
        
        if submitted:
            if email and name:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = name
                st.rerun()
            else:
                st.error("Please fill in all fields")

# ============================================================================
# AI CLIENT MANAGER
# ============================================================================

class AIClientManager:
    """Manage AI model connections"""
    
    def __init__(self):
        if not config.HF_TOKEN:
            st.error("⚠️ HF_TOKEN not configured. Please add your Hugging Face token to .env file")
            st.stop()
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

# ============================================================================
# SIMPLE DATABASE
# ============================================================================

class SimpleDatabase:
    """Simplified database for chat management"""
    
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Simple messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def add_message(self, chat_id: str, role: str, content: str):
        """Add message to database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        self.conn.commit()
    
    def get_messages(self, chat_id: str) -> List[dict]:
        """Get messages for a chat"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY timestamp",
            (chat_id,)
        )
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "timestamp": row[2]
            })
        return messages

# ============================================================================
# AI RESPONSE GENERATION
# ============================================================================

def generate_ai_response(prompt: str, context: str = "") -> Tuple[str, int, float]:
    """Generate AI response with error handling"""
    start_time = time.time()
    
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
            model=config.MODELS["primary"],
            max_tokens=config.MAX_TOKENS_PER_REQUEST,
            temperature=0.7,
            stream=False
        )
        
        content = response.choices[0].message.content
        tokens_used = len(content.split()) * 1.3  # Rough estimate
        processing_time = time.time() - start_time
        
        return content, int(tokens_used), processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\n\nError: {str(e)}"
        return error_msg, 0, processing_time

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="AMEK AI - Professional Code Generator",
    layout="wide",
    page_icon="🪄",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #131314;
        color: #E3E3E3;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E1F20 !important;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    
    .user-message {
        background-color: #2C2D2E;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #1E1F20;
        margin-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize
init_session_state()

# Initialize AI manager and database
@st.cache_resource
def get_ai_manager():
    return AIClientManager()

@st.cache_resource
def get_database():
    return SimpleDatabase()

ai_manager = get_ai_manager()
db = get_database()

# ============================================================================
# MAIN APP
# ============================================================================

if not st.session_state.authenticated:
    # Login screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🪄 AMEK AI")
        st.markdown("### Professional Code Generator")
        login_form()
        
        st.markdown("---")
        st.markdown("""
        **Features:**
        - 🚀 Advanced code generation
        - 💡 Intelligent problem solving  
        - 📚 Multi-language support
        - 🔒 Secure conversations
        
        **Supported Languages:**
        Python, JavaScript, TypeScript, Java, C++, Go, Rust, PHP, and more!
        """)

else:
    # Main app interface
    with st.sidebar:
        st.markdown("### 🪄 AMEK AI")
        st.caption("Professional Code Generator")
        
        # User info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"✨ **{st.session_state.user_name}**")
        with col2:
            if st.button("🚪", help="Logout"):
                st.session_state.authenticated = False
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()
        
        st.divider()
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_chat_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
    
    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = str(uuid.uuid4())[:8]
    
    # Display chat title
    st.markdown(f"### 💬 Chat {st.session_state.current_chat_id}")
    
    # Load messages from database if empty
    if not st.session_state.messages:
        st.session_state.messages = db.get_messages(st.session_state.current_chat_id)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about code, development, or technology..."):
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Save to database
        db.add_message(st.session_state.current_chat_id, "user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Build context from recent messages
                context = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-3:]  # Last 3 messages
                    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages[:-1]])
                
                response, tokens_used, processing_time = generate_ai_response(prompt, context)
                
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Save to database
                db.add_message(st.session_state.current_chat_id, "assistant", response)
                
                # Display response
                st.markdown(response)
                
                # Show processing info
                st.caption(f"⚡ {processing_time:.1f}s • 🎯 ~{tokens_used} tokens")
                
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9AA0A6; font-size: 12px; padding: 20px;">
    🪄 AMEK AI | Professional Code Generator<br>
    Built with ❤️ using Streamlit | Powered by Hugging Face
</div>
""", unsafe_allow_html=True)