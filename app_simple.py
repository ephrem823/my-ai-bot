import streamlit as st
import time

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("🪄 AMEK AI - Professional Code Generator")
    st.markdown("*Your intelligent coding companion*")
    
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
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                time.sleep(1)  # Simulate processing
                response = f"I received your request: {prompt}\n\nThis is a working minimal version. You can now integrate with your AI service."
                st.markdown(response)
                
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()