import streamlit as st
import bleach
import time
from typing import Optional, Dict

class Security:
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_user_info() -> Optional[Dict]:
        """Get current user information"""
        return st.session_state.get('user_info')
    
    @staticmethod
    def logout():
        """Clear user session"""
        keys_to_clear = ['authenticated', 'user_info', 'access_token']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        return bleach.clean(text, tags=[], attributes={}, strip=True)
    
    @staticmethod
    def is_admin(user_info: Dict) -> bool:
        """Check if user is admin"""
        admin_email = st.secrets.get("ADMIN_EMAIL") or "admin@example.com"
        return user_info.get('email') == admin_email