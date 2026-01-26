import streamlit as st
from huggingface_hub import InferenceClient
import concurrent.futures

# --- 1. CONFIGURATION ---
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}

st.set_page_config(page_title="ephrem AI SuperBrain", layout="wide")

# Custom CSS for the clean look and Sidebar History
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; min-width: 250px; }
    .stButton>button { width: 100%; text-align: left; border: none; background: transparent; color: #c9d1d9; }
    .stButton>button:hover { background: #21262d; color: #58a6ff; }
    .stApp { background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION INITIALIZATION ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} 
if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "New Chat"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR (HISTORY VIEW) ---
with st.sidebar:
    st.title("📜 History")
    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats[st.session_state.current_chat_title] = st.session_state.messages
        st.session_state.messages = []
        st.session_state.current_chat_title = f"Chat {len(st.session_state.all_chats) + 1}"
        st.rerun()
    st.divider()
    for title in list(st.session_state.all_chats.keys()):
        if st.button(f"💬 {title}"):
            st.session_state.current_chat_title = title
            st.session_state.messages = st.session_state.all_chats[title]
            st.rerun()

# --- 4. MAIN CHAT LOGIC ---
st.title(f"🧠 {st.session_state.current_chat_title}")

try:
    client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
except:
    st.error("Missing HF_TOKEN!")
    st.stop()

# Helper function for Parallel Calls
def call_ai(model_id, prompt_text, is_system=False):
    system_instr = "You are a code generator. Provide ONLY code and brief bulleted descriptions. No small talk."
    try:
        messages = [{"role": "system", "content": system_instr}] if is_system else []
        messages.append({"role": "user", "content": prompt_text})
        resp = client.chat.completions.create(model=model_id, messages=messages, max_tokens=1000)
        return resp.choices[0].message.content
    except: return ""

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="markdown")
        else:
            st.markdown(msg["content"])

# --- 5. CHAT INPUT ---
if prompt := st.chat_input("Hi or ask for code..."):
    # Rename chat on first message
    if not st.session_state.messages:
        st.session_state.current_chat_title = prompt[:25] + "..."
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # GREETING FILTER
        if prompt.lower().strip() in ["hi", "hello", "hey"]:
            final_answer = "hey"
            st.markdown(final_answer)
        else:
            # ENSEMBLE GENERATION
            with st.status("🧠 Generating and Merging Triple-Brain Response...") as status:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(call_ai, id, prompt, True) for id in MODELS.values()]
                    results = [f.result() for f in futures]
                
                # MERGE LOGIC
                merge_prompt = f"Combine these 3 code solutions into ONE best code block followed by a '### Logic Breakdown'. BE CONCISE. NO INTROS.\n1:{results[0]}\n2:{results[1]}\n3:{results[2]}"
                final_answer = call_ai(MODELS["deepseek"], merge_prompt)
                status.update(label="✅ Code Generated", state="complete")
            
            st.code(final_answer, language="markdown")
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.all_chats[st.session_state.current_chat_title] = st.session_state.messages
