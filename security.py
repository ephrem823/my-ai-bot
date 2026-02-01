"""Security middleware for Streamlit AI Bot"""

import streamlit as st
import hashlib
import time
import os
from functools import wraps
from typing import Dict, Any
import secrets

class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20"))
        self.max_tokens_per_request = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2500"))
        self.session_timeout = int(os.getenv("SESSION_TIMEOUT", "3600"))
        self.enable_security_headers = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"
        self.enable_csrf_protection = os.getenv("ENABLE_CSRF_PROTECTION", "true").lower() == "true"
        
    def init_security(self):
        """Initialize security settings"""
        if 'security_initialized' not in st.session_state:
            st.session_state.security_initialized = True
            st.session_state.request_count = {}
            st.session_state.last_activity = time.time()
            
    def check_session_timeout(self):
        """Check if session has timed out"""
        if 'last_activity' in st.session_state:
            if time.time() - st.session_state.last_activity > self.session_timeout:
                self.clear_session()
                return True
        return False
        
    def update_activity(self):
        """Update last activity timestamp"""
        st.session_state.last_activity = time.time()
        
    def rate_limit_check(self, user_id: str = "anonymous") -> bool:
        """Check rate limiting"""
        current_time = time.time()
        minute_key = int(current_time // 60)
        
        if 'request_count' not in st.session_state:
            st.session_state.request_count = {}
            
        user_requests = st.session_state.request_count.get(user_id, {})
        current_minute_requests = user_requests.get(minute_key, 0)
        
        if current_minute_requests >= self.max_requests_per_minute:
            return False
            
        # Update request count
        user_requests[minute_key] = current_minute_requests + 1
        st.session_state.request_count[user_id] = user_requests
        
        # Clean old entries
        cutoff_time = minute_key - 5  # Keep last 5 minutes
        for key in list(user_requests.keys()):
            if key < cutoff_time:
                del user_requests[key]
                
        return True
        
    def clear_session(self):
        """Clear session data"""
        keys_to_keep = ['security_initialized']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
                
    def sanitize_input(self, text: str) -> str:
        """Basic input sanitization"""
        if not isinstance(text, str):
            return ""
        # Remove potential XSS patterns
        dangerous_patterns = ['<script', '</script', 'javascript:', 'onload=', 'onerror=']
        for pattern in dangerous_patterns:
            text = text.replace(pattern.lower(), '')
            text = text.replace(pattern.upper(), '')
        return text.strip()
        
    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        if 'csrf_token' not in st.session_state:
            st.session_state.csrf_token = secrets.token_hex(32)
        return st.session_state.csrf_token
        
    def validate_csrf_token(self, token: str) -> bool:
        """Validate CSRF token"""
        return st.session_state.get('csrf_token') == token
    
    def validate_hf_input(self, text: str) -> bool:
        """Validate input for Hugging Face API"""
        if not text or len(text.strip()) == 0:
            return False
        
        # Check token limit
        word_count = len(text.split())
        if word_count > self.max_tokens_per_request:
            return False
        
        # Check for potential prompt injection
        dangerous_patterns = [
            'ignore previous', 'forget instructions', 'new instructions',
            'system prompt', 'override', 'jailbreak', 'pretend you are'
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                return False
        
        return True

def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from google_auth import is_logged_in, login_button
        
        if not is_logged_in():
            st.warning("🔒 Please log in to access this feature")
            login_button()
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def security_headers():
    """Add security headers"""
    st.markdown("""
    <script>
    // Disable right-click context menu
    document.addEventListener('contextmenu', function(e) {
        e.preventDefault();
    });
    
    // Disable F12, Ctrl+Shift+I, Ctrl+U
    document.addEventListener('keydown', function(e) {
        if (e.key === 'F12' || 
            (e.ctrlKey && e.shiftKey && e.key === 'I') ||
            (e.ctrlKey && e.key === 'u')) {
            e.preventDefault();
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Global security manager instance
security_manager = SecurityManager()