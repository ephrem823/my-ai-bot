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
    # Initial check, will be refined in main()
    oauth_available = True 
except Exception:
    google_oauth = None
    oauth_available = False

def display_message(message, is_user=False):
    """Display a chat message"""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        content = message["content"]
        
        if not is_user and "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part.strip():
                        st.markdown(part)
                else:
                    lines = part.split('\n')
                    language = lines[0] if lines[0] else "text"
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                    if code.strip():
                        st.code(code, language=language)
        else:
            st.markdown(content)
        
        if not is_user and message.get("tokens_used"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if message.get("model_used"):
                    st.caption(f"Model: {message['model_used']}")
            with col2:
                st.caption(f"Tokens: {message['tokens_used']}")
            with col3:
                if message.get("processing_time"):
                    st.caption(f"Time: {message['processing_time']:.1f}s")

def show_login_page():
    """Display login page with optional OAuth or guest access"""
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>AMEK AI</h1>
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
                            Sign in with Google
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.warning("Google login currently unavailable.")
        
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_info = {"name": "Guest User", "email": "guest@example.com"}
            st.rerun()
        
        if not oauth_available:
            st.info("OAuth not configured. Using guest mode.")

def main():
    """Main application function"""
    global oauth_available
    
    # 1. Check for HF token
    hf_token = st.secrets.get("HF_TOKEN") or getattr(config, 'HF_TOKEN', None)
    if not hf_token:
        st.error("⚠️ HF_TOKEN not found in secrets!")
        st.stop()
    
    # 2. Validate OAuth configuration from the [auth] section
    if "auth" in st.secrets:
        client_id = st.secrets["auth"].get("client_id")
        client_secret = st.secrets["auth"].get("client_secret")
        
        if client_id and client_secret:
            oauth_available = True
            os.environ["GOOGLE_CLIENT_ID"] = client_id
            os.environ["GOOGLE_CLIENT_SECRET"] = client_secret
        else:
            oauth_available = False
    else:
        oauth_available = False
    
    # Handle OAuth callback
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
    
    if not st.session_state.authenticated:
        show_login_page()
        return

    # Authenticated UI
    user_info = st.session_state.get("user_info", {"name": "User"})
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("AMEK AI")
        st.markdown("*Professional Code Generator*")
    with col2:
        st.markdown(f"**Hi, {user_info.get('name', 'User')}!**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        selected_model = st.selectbox(
            "AI Model",
            options=list(config.MODELS.values()),
            index=0
        )

    # Chat
    for message in st.session_state.messages:
        display_message(message, is_user=(message["role"] == "user"))
    
    if prompt := st.chat_input("Ask me anything..."):
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_message(user_message, is_user=True)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
