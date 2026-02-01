"""Secure AI Bot with Streamlit Security Features"""

import streamlit as st
import os
from dotenv import load_dotenv
from security import security_manager, require_auth, security_headers
from google_auth import init_google_auth, is_logged_in, get_user, login_button, logout
from hf_secure import hf_client
import requests

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Secure AI Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application with security"""
    
    # Initialize security
    security_manager.init_security()
    
    # Check session timeout
    if security_manager.check_session_timeout():
        st.warning("⏰ Session expired. Please log in again.")
        st.stop()
    
    # Update activity
    security_manager.update_activity()
    
    # Add security headers
    security_headers()
    
    # Initialize Google Auth
    init_google_auth()
    
    # Header
    st.title("🔒 Secure AI Bot")
    
    # Authentication section
    if not is_logged_in():
        st.markdown("### Welcome to Secure AI Bot")
        st.info("Please log in to access the AI chat features.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button()
        
        st.markdown("---")
        st.markdown("#### Security Features:")
        st.markdown("""
        - 🔐 Google OAuth Authentication
        - 🛡️ Rate Limiting Protection
        - 🔒 Session Management
        - 🚫 CSRF Protection
        - 🔍 Input Sanitization
        - ⏰ Session Timeout
        """)
        return
    
    # User is logged in
    user = get_user()
    
    # Sidebar with user info and security status
    with st.sidebar:
        st.markdown("### User Profile")
        if user.get('picture'):
            st.image(user['picture'], width=80)
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Email:** {user['email']}")
        
        if st.button("🚪 Logout"):
            logout()
        
        st.markdown("---")
        st.markdown("### Security Status")
        st.success("✅ Authenticated")
        st.info(f"🔒 Session Active")
        
        # Rate limit status
        user_id = user['email']
        if security_manager.rate_limit_check(user_id):
            st.success("✅ Rate Limit OK")
        else:
            st.error("⚠️ Rate Limited")
    
    # Main chat interface
    st.markdown("### AI Chat Interface")
    
    # Rate limiting check
    if not security_manager.rate_limit_check(user['email']):
        st.error("⚠️ Rate limit exceeded. Please wait a moment before sending another message.")
        return
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with security
    if prompt := st.chat_input("What would you like to know?"):
        # Validate input for Hugging Face
        if not security_manager.validate_hf_input(prompt):
            st.error("⚠️ Invalid input detected. Please check your message for length and content.")
            return
        
        # Sanitize input
        prompt = security_manager.sanitize_input(prompt)
        
        if not prompt:
            st.error("Invalid input detected.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response (placeholder)
        with st.chat_message("assistant"):
            response = generate_ai_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

@require_auth
def generate_ai_response(prompt: str) -> str:
    """Generate AI response with security checks"""
    from hf_secure import hf_client
    
    try:
        # Generate response using Hugging Face
        response = hf_client.generate_text(prompt)
        
        if response:
            return response
        else:
            return "I'm having trouble processing your request right now. Please try again."
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm unable to process your request at the moment."

def admin_panel():
    """Admin panel for security monitoring"""
    user = get_user()
    admin_email = os.getenv("ADMIN_EMAIL")
    
    if user and user['email'] == admin_email:
        st.markdown("### 🛡️ Admin Security Panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active Sessions", len(st.session_state.get('request_count', {})))
            st.metric("Security Events", 0)  # Placeholder
        
        with col2:
            st.metric("Rate Limited Users", 0)  # Placeholder
            st.metric("Failed Logins", 0)  # Placeholder
        
        if st.button("Clear All Sessions"):
            security_manager.clear_session()
            st.success("All sessions cleared!")

if __name__ == "__main__":
    main()
    
    # Show admin panel if user is admin
    user = get_user()
    if user and user['email'] == os.getenv("ADMIN_EMAIL"):
        st.markdown("---")
        admin_panel()