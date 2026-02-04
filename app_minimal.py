import streamlit as st
import time
from typing import Optional, Tuple

# Import your custom modules
import config
import google_oauth
import security

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ============================================================================
# AI RESPONSE GENERATION
# ============================================================================

def generate_ai_response(prompt: str, context: str = "", model: str = None) -> Tuple[str, int, float]:
    """Generate AI response with error handling and fallback"""
    start_time = time.time()
    
    try:
        # Simple response for now - replace with actual AI integration
        response_content = f"I received your request: {prompt}\n\nThis is a placeholder response. Please integrate with your import streamlit as st
import time
import requests
from typing import Dict, List, Optional

# Import local modules
import config
from security import Security
from google_oauth import GoogleOAuth

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize components
security = Security()
google_oauth = GoogleOAuth()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# AI RESPONSE FUNCTIONS
# ============================================================================

def generate_ai_response(prompt: str, model: str = "codellama/CodeLlama-7b-Instruct-hf") -> tuple[str, int, float]:
    """Generate AI response using Hugging Face API"""
    start_time = time.time()
    
    try:
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {config.HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create a focused prompt for code generation
        system_prompt = "You are a professional code assistant. Provide clear, efficient, and well-documented code solutions."
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Make API request
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                response_content = result[0].get("generated_text", "")
            else:
                response_content = "I apologize, but I couldn't generate a response. Please try again."
        else:
            response_content = f"I'm currently experiencing high demand. Please try again in a moment. (Status: {response.status_code})"
        
        processing_time = time.time() - start_time
        return response_content, 50, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\n\nError: {str(e)}"
        return error_msg, 0, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\n\nError: {str(e)}"
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
    """Display login page with Google OAuth or skip authentication"""
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🪄 AMEK AI</h1>
        <h3>Professional Code Generator</h3>
        <p style="color: #888; margin-bottom: 40px;">Please sign in with your Google account to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
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
            
        except ValueError as e:
            st.warning(f"⚠️ OAuth not configured: {str(e)}")
            st.info("""
            **OAuth is not configured. You can:**
            1. **Continue without authentication** (click button below)
            2. **Set up Google OAuth** by following these steps:
               - Go to [Google Cloud Console](https://console.cloud.google.com/)
               - Create OAuth 2.0 credentials
               - Update your `.env` file with real credentials
            """)
            
            if st.button("🚀 Continue Without Authentication", use_container_width=True):
                # Set up a basic session without OAuth
                st.session_state.authenticated = True
                st.session_state.user_info = {"name": "Guest User", "email": "guest@example.com"}
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