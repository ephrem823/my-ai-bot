import streamlit as st
from huggingface_hub import InferenceClient
from authlib.integrations.requests_client import OAuth2Session
import time

# Configuration
HF_TOKEN = st.secrets["HF_TOKEN"]
GOOGLE_CLIENT_ID = st.secrets["auth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["auth"]["client_secret"]
REDIRECT_URI = st.secrets["auth"]["redirect_uri"]

# Initialize
client = InferenceClient(api_key=HF_TOKEN)

st.set_page_config(page_title="AMEK AI", page_icon="🪄", layout="wide")

# CSS
st.markdown("""
<style>
.stApp { background-color: #131314; color: #E3E3E3; }
[data-testid="stSidebar"] { background-color: #1E1F20 !important; }
.stChatInputContainer { border-radius: 32px !important; background-color: #1E1F20 !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

def google_login():
    """Initiate Google OAuth"""
    oauth = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        scope="openid email profile",
        redirect_uri=REDIRECT_URI
    )
    authorization_url, state = oauth.create_authorization_url(
        "https://accounts.google.com/o/oauth2/v2/auth"
    )
    st.session_state.oauth_state = state
    return authorization_url

def handle_oauth_callback():
    """Handle OAuth callback"""
    query_params = st.query_params
    if "code" in query_params:
        oauth = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            state=st.session_state.get("oauth_state")
        )
        token = oauth.fetch_token(
            "https://oauth2.googleapis.com/token",
            code=query_params["code"]
        )
        user_info = oauth.get("https://www.googleapis.com/oauth2/v1/userinfo").json()
        st.session_state.authenticated = True
        st.session_state.user_email = user_info.get("email")
        st.session_state.user_name = user_info.get("name")
        st.query_params.clear()
        st.rerun()

def generate_response(prompt):
    """Generate AI response"""
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are AMEK, a professional AI assistant."},
                {"role": "user", "content": prompt}
            ],
            model="deepseek-ai/DeepSeek-V3",
            max_tokens=2500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Handle OAuth callback
handle_oauth_callback()

# Main UI
st.title("🪄 AMEK AI")

# Sidebar
with st.sidebar:
    if st.session_state.authenticated:
        st.write(f"👤 {st.session_state.user_name}")
        st.write(f"📧 {st.session_state.user_email}")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()
    
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Check if authenticated
    if not st.session_state.authenticated:
        st.session_state.pending_prompt = prompt
        auth_url = google_login()
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3>Sign in to continue</h3>
            <p>Please sign in with your Google account to use AMEK AI</p>
            <a href="{auth_url}" target="_self">
                <button style="background: #4285f4; color: white; padding: 12px 24px; 
                border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">
                    Sign in with Google
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Process pending prompt after login
if st.session_state.authenticated and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
