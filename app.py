import streamlit as st
import os
from dotenv import load_dotenv
import config
import security
from ai_service import generate_ai_response

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Try to initialize OAuth, but don't fail if not configured
try:
    from google_oauth import GoogleOAuth
    google_oauth = GoogleOAuth()
    oauth_available = google_oauth.is_configured
except Exception:
    oauth_available = False

def display_message(message, is_user=False):
    """Display a chat message"""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        content = message["content"]
        
        if not is_user and "```" in content:
            # Handle code blocks
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
        
        # Show metadata for assistant messages
        if not is_user and message.get("tokens_used"):
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

def show_login_page():
    """Display login page with optional OAuth or guest access"""
    global oauth_available
    
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🪄 AMEK AI</h1>
        <h3>Professional Code Generator</h3>
        <p style="color: #888; margin-bottom: 40px;">Your intelligent coding companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if oauth_available:
            try:
                auth_url = google_oauth.get_auth_url()
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <a href="{auth_url}" target="_self">
                        <button style="
                            background-color: #4285f4;
                            color: white;
                            border: none;
                            padding: 12px 24px;
                            border-radius: 8px;
                            font-size: 16px;
                            cursor: pointer;
                            width: 100%;
                        ">
                            🔐 Sign in with Google
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                oauth_available = False
        
        # Always show guest option
        if st.button("🚀 Continue as Guest", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_info = {"name": "Guest User", "email": "guest@example.com"}
            st.rerun()
        
        if not oauth_available:
            st.info("💡 OAuth not configured. Using guest mode.")

def main():
    """Main application function"""
    
    # Check for HF token
    hf_token = st.secrets.get("HF_TOKEN") or config.HF_TOKEN
    if not hf_token or hf_token == "your_huggingface_token_here":
        st.error("⚠️ Please configure HF_TOKEN in Streamlit secrets or .env file")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        st.stop()
    
    # Handle OAuth callback if available
    if oauth_available:
        query_params = st.query_params
        if "code" in query_params:
            try:
                token_data = google_oauth.exchange_code_for_token(query_params["code"])
                if "access_token" in token_data:
                    user_info = google_oauth.get_user_info(token_data["access_token"])
                    st.session_state.authenticated = True
                    st.session_state.user_info = user_info
                    st.query_params.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Main app interface
    user_info = st.session_state.get("user_info", {"name": "User"})
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🪄 AMEK AI - Professional Code Generator")
        st.markdown("*Your intelligent coding companion*")
    with col2:
        st.markdown(f"**Welcome, {user_info.get('name', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_info = None
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