import streamlit as st
from huggingface_hub import InferenceClient

# - CONFIGURATION & MODELS -
MODELS = {
    "Qwen 2.5 Coder (Best All-Rounder)": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "DeepSeek V3.2 (Advanced Logic)": "deepseek-ai/DeepSeek-V3.2",
    "GLM 4.7 Flash (Fast & Precise)": "zai-org/GLM-4.7-Flash"
}

# --- PAGE SETUP ---
st.set_page_config(page_title="amek AI Assistant", layout="wide")
st.title("🤖 ephrem AI Coding Assistant")

# --- SECRETS CHECK ---
try:
    MY_HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ HF_TOKEN not found! Please add it to Streamlit Secrets.")
    st.stop()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selection Dropdown
    selected_model_display = st.selectbox("Choose AI Brain:", list(MODELS.keys()))
    MODEL_NAME = MODELS[selected_model_display]
    
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.info(f"Currently using: \n**{selected_model_display}**")

# --- INITIALIZE CHAT ---
client = InferenceClient(api_key=MY_HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with automatic Copy Buttons
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Using .code() ensures the entire message has a copy button
            st.code(message["content"], language="markdown")
        else:
            st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask me to write or debug code..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # System instructions to keep the AI focused
        messages_for_api = [
            {"role": "system", "content": "You are an expert senior software engineer. Provide clean, efficient code with clear step-by-step explanations."}
        ]
        
        # Add history for context
        for m in st.session_state.messages:
            messages_for_api.append({"role": m["role"], "content": m["content"]})

        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_api,
                max_tokens=2000,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # Display markdown stream while typing
                        response_placeholder.markdown(full_response + "▌")
            
            # Finalize with st.code to add the COPY BUTTON
            response_placeholder.code(full_response, language="markdown")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"API Error: {e}. Try switching to a different model in the sidebar.")
