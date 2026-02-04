import streamlit as st
import time
import asyncio
import concurrent.futures
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor

# Import modules
import config
from security import Security
from google_oauth import GoogleOAuth

# Initialize components
security = Security()
google_oauth = GoogleOAuth()

# Page config
st.set_page_config(
    page_title="AMEK AI - Code Generator",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .quick-action-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        cursor: pointer;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {
        "theme": "light",
        "model": "codellama",
        "max_tokens": 1000,
        "temperature": 0.7
    }

class AIManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.cache = {}
    
    def generate_response_async(self, prompt: str, model: str, **kwargs) -> Tuple[str, int, float]:
        """Generate AI response with async processing"""
        cache_key = f"{prompt}_{model}_{kwargs.get('temperature', 0.7)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Simulate API call with threading
            future = self.executor.submit(self._make_api_call, prompt, model, **kwargs)
            response = future.result(timeout=30)
            
            processing_time = time.time() - start_time
            tokens_used = len(prompt.split()) + len(response.split())
            
            result = (response, tokens_used, processing_time)
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            return error_msg, 0, processing_time
    
    def _make_api_call(self, prompt: str, model: str, **kwargs) -> str:
        """Make actual API call to Hugging Face"""
        headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False
            }
        }
        
        model_url = f"https://api-inference.huggingface.co/models/{model}"
        response = requests.post(model_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            return str(result)
        else:
            raise Exception(f"API Error: {response.status_code}")

ai_manager = AIManager()

def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.messages:
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "user_info": security.get_user_info()
        }
        return json.dumps(chat_data, indent=2)
    return None

def import_chat_history(uploaded_file):
    """Import chat history from JSON file"""
    try:
        chat_data = json.load(uploaded_file)
        if "messages" in chat_data:
            st.session_state.messages = chat_data["messages"]
            st.success("Chat history imported successfully!")
            st.rerun()
    except Exception as e:
        st.error(f"Error importing chat history: {str(e)}")

def display_message_enhanced(message: dict, is_user: bool = False):
    """Enhanced message display with better formatting"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.markdown(f"**You:** {message['content']}")
        else:
            content = message["content"]
            
            # Enhanced code block detection and highlighting
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
                            col1, col2 = st.columns([10, 1])
                            with col1:
                                st.code(code, language=language)
                            with col2:
                                if st.button("📋", key=f"copy_{hash(code)}", help="Copy code"):
                                    st.write("Copied!")
            else:
                st.markdown(content)
            
            # Enhanced metadata display
            if message.get("tokens_used") or message.get("processing_time"):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    if message.get("model_used"):
                        st.caption(f"🤖 {message['model_used']}")
                with col2:
                    if message.get("tokens_used"):
                        st.caption(f"📊 {message['tokens_used']} tokens")
                with col3:
                    if message.get("processing_time"):
                        st.caption(f"⏱️ {message['processing_time']:.1f}s")
                with col4:
                    timestamp = message.get("timestamp", datetime.now().strftime("%H:%M"))
                    st.caption(f"🕒 {timestamp}")

def show_analytics_dashboard():
    """Display analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.messages:
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        ai_messages = total_messages - user_messages
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("User Messages", user_messages)
        with col3:
            st.metric("AI Responses", ai_messages)
        with col4:
            avg_response_time = sum([m.get("processing_time", 0) for m in st.session_state.messages]) / max(ai_messages, 1)
            st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
        
        # Token usage chart
        tokens_data = [m.get("tokens_used", 0) for m in st.session_state.messages if m.get("tokens_used")]
        if tokens_data:
            st.line_chart(tokens_data)
    else:
        st.info("No chat data available yet. Start a conversation to see analytics!")

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
    """Enhanced login page"""
    st.markdown("""
    <div class="main-header" style="text-align: center;">
        <h1>🪄 AMEK AI</h1>
        <h3>Professional Code Generator</h3>
        <p>Advanced AI-powered coding assistant with enhanced features</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        auth_url = google_oauth.get_auth_url()
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="{auth_url}" target="_self">
                <button class="quick-action-btn" style="
                    background: linear-gradient(45deg, #4285f4, #34a853);
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 25px;
                    font-size: 18px;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    text-decoration: none;
                    box-shadow: 0 4px 15px rgba(66, 133, 244, 0.3);
                ">
                    <svg width="24" height="24" viewBox="0 0 24 24">
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
    """Enhanced main application"""
    
    if not config.HF_TOKEN:
        st.error("⚠️ HF_TOKEN not found in .env file. Please add your Hugging Face API token.")
        st.stop()
    
    handle_oauth_callback()
    
    if not security.is_authenticated():
        show_login_page()
        return
    
    user_info = security.get_user_info()
    
    # Enhanced header
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>🪄 AMEK AI - Enhanced Code Generator</h1>
                <p>Advanced AI-powered coding assistant</p>
            </div>
            <div style="text-align: right;">
                <h3>Welcome, {user_info.get('name', 'User')}!</h3>
                <p>{user_info.get('email', '')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("🎛️ Controls")
        
        # Enhanced controls
        tab1, tab2, tab3 = st.tabs(["Chat", "Settings", "Analytics"])
        
        with tab1:
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("💾 Export Chat", use_container_width=True):
                chat_export = export_chat_history()
                if chat_export:
                    st.download_button(
                        "📥 Download",
                        chat_export,
                        f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
            
            uploaded_file = st.file_uploader("📤 Import Chat", type="json")
            if uploaded_file:
                import_chat_history(uploaded_file)
            
            if st.button("🚪 Logout", use_container_width=True):
                security.logout()
                st.rerun()
        
        with tab2:
            st.subheader("Model Settings")
            selected_model = st.selectbox(
                "🤖 AI Model",
                options=list(config.MODELS.keys()),
                index=0
            )
            
            temperature = st.slider("🌡️ Temperature", 0.1, 1.0, 0.7, 0.1)
            max_tokens = st.slider("📏 Max Tokens", 100, 2000, 1000, 100)
            
            st.subheader("Quick Actions")
            quick_prompts = [
                "Create a Python function",
                "Debug this code",
                "Optimize performance", 
                "Add error handling",
                "Write unit tests",
                "Explain this algorithm",
                "Convert to different language",
                "Add documentation"
            ]
            
            for prompt in quick_prompts:
                if st.button(prompt, use_container_width=True, key=f"quick_{prompt}"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.rerun()
        
        with tab3:
            show_analytics_dashboard()
    
    with col1:
        st.header("💬 Chat")
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message_enhanced(message, is_user=(message["role"] == "user"))
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about coding..."):
            prompt = security.sanitize_input(prompt)
            
            # Add timestamp to user message
            user_message = {
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.messages.append(user_message)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Generating response..."):
                    response_content, tokens_used, processing_time = ai_manager.generate_response_async(
                        prompt,
                        config.MODELS[selected_model],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "tokens_used": tokens_used,
                        "processing_time": processing_time,
                        "model_used": selected_model,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    
                    st.session_state.messages.append(assistant_message)
                    display_message_enhanced(assistant_message)

if __name__ == "__main__":
    main()