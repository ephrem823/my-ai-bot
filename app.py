import streamlit as st
from huggingface_hub import InferenceClient

# --- CONFIGURATION ---
# We use st.secrets so your token stays hidden from GitHub!
try:
    MY_HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Error: HF_TOKEN not found in Streamlit Secrets!")
    st.stop()

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"

st.set_page_config(page_title="amek AI", layout="centered")
st.title("🤖 ephrem AI Assistant")

client = InferenceClient(api_key=MY_HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your AI anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.st.code(full_response, language="markdown")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"API Error: {e}")
