import streamlit as st
import time
import datetime
import uuid
import sqlite3
import os
from huggingface_hub import InferenceClient
import streamlit_authenticator as stauth
from google.oauth2 import id_token
from google.auth.transport import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODELS = {"primary": "microsoft/DialoGPT-medium"}
    MAX_TOKENS_PER_REQUEST = 1000
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
    GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", "")

config = Config()

# ============================================================================
# DATABASE
# ============================================================================

class SimpleDatabase:
    def __init__(self):
        self.db_path = "chat_history.db"
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def add_message(self, chat_id, role, content):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, datetime.datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    
    def get_messages(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp",
            (chat_id,)
        )
        messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        conn.close()
        return messages

# ============================================================================
# AI CLIENT
# ============================================================================

class AIClientManager:
    def __init__(self):
        self.client = InferenceClient(token=config.HF_TOKEN) if config.HF_TOKEN else None
    
    def chat_completion(self, messages, model, max_tokens, temperature, stream):
        if not self.client:
            raise Exception("HF_TOKEN not configured")
        
        # Simple response generation
        prompt = messages[-1]["content"]
        response = self.client.text_generation(
            prompt=f"User: {prompt}\nAssistant:",
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # Mock response structure
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
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

# ============================================================================
# AUTHENTICATION
# ============================================================================

def login_form():
    st.markdown("### Login with Google")
    
    # Google Sign-In button HTML
    google_signin_html = f"""
    <div id="g_id_onload"
         data-client_id="{config.GOOGLE_CLIENT_ID}"
         data-callback="handleCredentialResponse">
    </div>
    <div class="g_id_signin" data-type="standard"></div>
    
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    <script>
    function handleCredentialResponse(response) {{
        // Send the credential to Streamlit
        window.parent.postMessage({{
            type: 'google_auth',
            credential: response.credential
        }}, '*');
    }}
    </script>
    """
    
    st.components.v1.html(google_signin_html, height=100)
    
    st.markdown("---")
    st.markdown("### Or login manually")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        name = st.text_input("Name")
        
        if st.form_submit_button("Login"):
            if email and name:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = name
                st.rerun()
            else:
                st.error("Please fill in all fields")

# ============================================================================
# AI RESPONSE GENERATION
# ============================================================================

def generate_ai_response(prompt, context=""):
    start_time = time.time()
    
    try:
        client = AIClientManager()
        
        messages = [
            {"role": "system", "content": """You are AMEK AI, a professional code generator and programming assistant.
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
        tokens_used = len(content.split()) * 1.3
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
def get_database():
    return SimpleDatabase()

db = get_database()

# ============================================================================
# MAIN APPLICATION
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
                    recent_messages = st.session_state.messages[-3:]
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9AA0A6; font-size: 12px; padding: 20px;">
    🪄 AMEK AI | Professional Code Generator<br>
    Built with ❤️ using Streamlit | Powered by Hugging Face
</div>
""", unsafe_allow_html=True)