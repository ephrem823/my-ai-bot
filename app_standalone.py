import streamlit as st
import time

# Simple standalone AI bot without external dependencies

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
            st.session_state.authenticated = True
            st.rerun()

def main():
    """Main application function"""
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🪄 AMEK AI - Professional Code Generator")
        st.markdown("*Your intelligent coding companion*")
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.info("💡 This is a demo version. Connect your AI service for full functionality.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about coding..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate demo response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                time.sleep(1)  # Simulate processing
                
                # Simple demo responses
                if "python" in prompt.lower():
                    response = "Here's a Python example:\n\n```python\ndef hello_world():\n    print('Hello, World!')\n    return 'Success'\n\nhello_world()\n```"
                elif "javascript" in prompt.lower():
                    response = "Here's a JavaScript example:\n\n```javascript\nfunction helloWorld() {\n    console.log('Hello, World!');\n    return 'Success';\n}\n\nhelloWorld();\n```"
                else:
                    response = f"I understand you're asking about: '{prompt}'\n\nThis is a demo version. To get real AI responses, please configure your AI service in the full version."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()