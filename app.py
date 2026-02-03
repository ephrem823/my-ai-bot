import streamlit as st
import os
import time
import uuid
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ HUGGINGFACE_API_TOKEN not found in .env file")
    st.stop()

# Initialize client
client = InferenceClient(token=HF_TOKEN)

# Page config
st.set_page_config(
    page_title="AMEK AI - Code Generator",
    page_icon="🪄",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def generate_response(prompt):
    """Generate AI response"""
    try:
        messages = [
            {"role": "system", "content": "You are AMEK, a professional AI code generator. Provide high-quality, well-documented code solutions."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
            model="deepseek-ai/DeepSeek-V3",
            max_tokens=2500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Authentication
if not st.session_state.authenticated:
    st.title("🪄 AMEK AI - Code Generator")
    
    with st.form("login"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        submit = st.form_submit_button("Login")
        
        if submit and name and email:
            st.session_state.authenticated = True
            st.session_state.user_name = name
            st.rerun()

else:
    # Main app
    st.title("🪄 AMEK AI - Code Generator")
    st.markdown(f"Welcome, **{st.session_state.user_name}**!")
    
    # Sidebar
    with st.sidebar:
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about coding..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})