import streamlit as st
import time
import requests
from typing import Dict, List, Optional

# Import local modules
import config
from security import Security
from ai_service import generate_ai_response

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize components
security = Security()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

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
                        lines = part.split('\n')
                        language = lines[0] if lines[0] else "text"
                        code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                        
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

def show_login_page():
    """Display simple login page"""
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🪄 AMEK AI</h1>
        <h3>Professional Code Generator</h3>
        <p style="color: #888; margin-bottom: 40px;">Click below to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Continue", use_container_width=True):
            # Set up a basic session
            st.session_state.authenticated = True
            st.session_state.user_info = {"name": "User", "email": "user@example.com"}
            st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Check for required environment variables using Streamlit secrets
    hf_token = st.secrets.get("HF_TOKEN") or config.HF_TOKEN
    if not hf_token:
        st.error("⚠️ HF_TOKEN not found in Streamlit secrets or .env file.")
        st.stop()
    
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