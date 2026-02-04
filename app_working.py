import streamlit as st
import time
from typing import Tuple

# Page configuration
st.set_page_config(
    page_title="AMEK AI - Code Generator",
    page_icon="🪄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_ai_response(prompt: str) -> Tuple[str, int, float]:
    """Generate AI response with placeholder logic"""
    start_time = time.time()
    
    try:
        # Simulate processing time
        time.sleep(1)
        
        # Simple response logic based on keywords
        if "python" in prompt.lower():
            response = f"""Here's a Python solution for your request:

```python
def example_function():
    # This is a placeholder response
    # Replace with actual AI integration
    print("Hello from AMEK AI!")
    return "Success"

# Usage example
result = example_function()
print(result)
```

This is a placeholder response. To get real AI-powered responses, you'll need to:
1. Add your Hugging Face API token to the .env file
2. Or integrate with OpenAI, Claude, or another AI service
"""
        elif "javascript" in prompt.lower() or "js" in prompt.lower():
            response = f"""Here's a JavaScript solution:

```javascript
function exampleFunction() {
    // This is a placeholder response
    console.log("Hello from AMEK AI!");
    return "Success";
}

// Usage example
const result = exampleFunction();
console.log(result);
```

This is a placeholder response. Please configure your AI service for real responses.
"""
        else:
            response = f"""I received your request: "{prompt}"

This is a placeholder response. To enable real AI responses:

1. **Option 1 - Hugging Face (Free)**:
   - Get a free API token from https://huggingface.co/settings/tokens
   - Add it to your .env file: `HF_TOKEN=your_actual_token_here`

2. **Option 2 - OpenAI**:
   - Get an API key from https://platform.openai.com/api-keys
   - Integrate OpenAI API in the code

3. **Option 3 - Local AI**:
   - Use Ollama or similar local AI models

For now, I can help with basic coding questions using placeholder responses!
"""
        
        processing_time = time.time() - start_time
        return response, 50, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error generating response: {str(e)}"
        return error_msg, 0, processing_time

def display_message(message: dict, is_user: bool = False):
    """Display a chat message with proper formatting"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.markdown(message["content"])
        else:
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
                with col2:
                    if message.get("tokens_used"):
                        st.caption(f"📊 {message['tokens_used']} tokens")
                with col3:
                    if message.get("processing_time"):
                        st.caption(f"⏱️ {message['processing_time']:.1f}s")

def main():
    """Main application function"""
    
    # Header
    st.title("🪄 AMEK AI - Professional Code Generator")
    st.markdown("*Your intelligent coding companion (Demo Mode)*")
    
    # Info about demo mode
    with st.expander("ℹ️ Demo Mode Information"):
        st.info("""
        **You're running in Demo Mode!**
        
        This version works without external API keys. To enable full AI capabilities:
        
        1. **Get a Hugging Face token** (free): https://huggingface.co/settings/tokens
        2. **Update your .env file** with real credentials
        3. **Restart the app**
        
        Current features in demo mode:
        - ✅ Basic chat interface
        - ✅ Code syntax highlighting  
        - ✅ Placeholder AI responses
        - ❌ Real AI model integration
        - ❌ Google OAuth authentication
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Model selection (demo)
        st.selectbox(
            "🤖 AI Model (Demo)",
            options=["Demo Mode - Placeholder Responses"],
            index=0,
            disabled=True
        )
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        quick_prompts = [
            "Create a Python function",
            "Write JavaScript code",
            "Debug this code",
            "Optimize performance",
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
                response_content, tokens_used, processing_time = generate_ai_response(prompt)
                
                # Create assistant message with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "tokens_used": tokens_used,
                    "processing_time": processing_time
                }
                
                st.session_state.messages.append(assistant_message)
                display_message(assistant_message)

if __name__ == "__main__":
    main()