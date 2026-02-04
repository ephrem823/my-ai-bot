import streamlit as st
import time
from google_oauth import GoogleOAuth
from security import Security
import config

# Page config
st.set_page_config(page_title="AMEK AI", page_icon="🪄", layout="wide")

# Initialize components
google_oauth = GoogleOAuth()
security = Security()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_ai_response(prompt: str, model: str = None) -> tuple:
    """Generate AI response (placeholder implementation)"""
    # Your existing AI response logic here
    return "AI response placeholder", 100, 1.5

def display_message(message: dict, is_user: bool = False):
    """Display a chat message with proper formatting"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.markdown(message["content"])
        else:
            content = message["content"]
            
            if "```" in content:
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

def handle_oauth_callback():
    """Handle OAuth callback from Google"""
    query_params = st.query_params
    
    if "code" in query_params:
        try:
            token_data = google_oauth.exchange_code_for_token(query_params["code"])
            
            if "access_token" in token_data:
                user_info = google_oauth.get_user_info(token_data["access_token"])
                
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.session_state.access_token = token_data["access_token"]
                
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

def main():
    """Main application function"""
    
    if not config.HF_TOKEN:
        st.error("⚠️ HF_TOKEN not found in .env file. Please add your Hugging Face API token.")
        st.stop()
    
    handle_oauth_callback()
    
    if not security.is_authenticated():
        show_login_page()
        return
    
    user_info = security.get_user_info()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🪄 AMEK AI - Professional Code Generator")
        st.markdown("*Your intelligent coding companion*")
    with col2:
        st.markdown(f"**Welcome, {user_info.get('name', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            security.logout()
            st.rerun()
    
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        selected_model = st.selectbox(
            "🤖 AI Model",
            options=list(config.MODELS.values()),
            index=0
        )
        
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
    
    for message in st.session_state.messages:
        display_message(message, is_user=(message["role"] == "user"))
    
    if prompt := st.chat_input("Ask me anything about coding..."):
        prompt = security.sanitize_input(prompt)
        
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_message(user_message, is_user=True)
        
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