import streamlit as st
import os
import datetime
import concurrent.futures
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. SETTINGS & AI MODELS ---
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}
ADMIN_EMAIL = "efaxalemayehu@gmail.com"

# --- 2. GEMINI THEME UI (CSS) ---
st.set_page_config(page_title="AMEK", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* Gemini Dark Mode Palette */
    .stApp { background-color: #131314; color: #e3e3e3; }
    [data-testid="stSidebar"] { background-color: #1e1f20 !important; border-right: none; }
    
    /* Center Login Page */
    .login-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh; }
    
    /* Chat Bubbles */
    .stChatMessage { background-color: transparent !important; border: none !important; }
    .stCodeBlock { border-radius: 12px; border: 1px solid #3c4043; }
    
    /* Fast Input Bar */
    .stChatInputContainer { border-radius: 28px !important; border: 1px solid #3c4043 !important; background-color: #1e1f20 !important; }
    
    /* Hide specific Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIN GATE (GEMINI STYLE) ---
if not st.user.is_logged_in:
    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.write("#")
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>AMEK</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8e918f;'>Hello. How can I help you code today?</p>", unsafe_allow_html=True)
        st.write("#")
        st.button("✨ Sign in with Google", on_click=st.login, use_container_width=True, type="primary")
    st.stop()

# --- 4. ENGINE SETUP (CACHED FOR SPEED) ---
@st.cache_resource
def get_engine():
    return SentenceTransformer("all-MiniLM-L6-v2")

class FastEmbed:
    def __init__(self, m): self.m = m
    def embed_documents(self, t): return self.m.encode(t).tolist()
    def embed_query(self, t): return self.m.encode([t])[0].tolist()

raw_m = get_engine()
embedder = FastEmbed(raw_m)

# --- 5. LOGGING & SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ AMEK Settings")
    st.write(f"Account: **{st.user.email}**")
    if st.button("Logout"): st.logout()
    
    if st.user.email == ADMIN_EMAIL:
        with st.expander("🔐 Usage Logs"):
            if os.path.exists("usage_logs.txt"):
                with open("usage_logs.txt", "r") as f: st.text(f.read())
                if st.button("Clear Logs"): 
                    os.remove("usage_logs.txt")
                    st.rerun()

# --- 6. CHAT LOGIC ---
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

def stream_ai(m_id, prompt, context=""):
    sys = "You are AMEK. Strict Code Assistant. If non-code, say 'this ai is only for code except hey'."
    full = f"Context: {context}\n\nTask: {prompt}" if context else prompt
    msg = [{"role": "system", "content": sys}, {"role": "user", "content": full}]
    return client.chat.completions.create(model=m_id, messages=msg, stream=True, max_tokens=2000)

if "messages" not in st.session_state: st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "this ai" not in m["content"]: st.code(m["content"])
        else: st.markdown(m["content"])

# --- 7. UNIFIED INPUT (GEMINI INPUT BAR) ---
user_input = st.chat_input("Enter a prompt here...", accept_file=True, file_type=["pdf", "py", "txt"])

if user_input:
    prompt, files = user_input.text, user_input.files
    
    # Process Files (Background)
    ctx = ""
    if files:
        for f in files:
            if f.type == "application/pdf":
                with open("t.pdf", "wb") as tmp: tmp.write(f.getvalue())
                ctx += "\n".join([d.page_content for d in PyPDFLoader("t.pdf").load()])
            else: ctx += f.read().decode("utf-8")

    st.session_state.messages.append({"role": "user", "content": prompt or "Uploaded Files"})
    with st.chat_message("user"): st.markdown(prompt or "📎 Files attached")

    with st.chat_message("assistant"):
        # Instant Response for 'hey'
        if prompt and prompt.lower().strip() == "hey":
            ans = "hey"
            st.markdown(ans)
        else:
            # STREAMING EFFECT LIKE GEMINI
            placeholder = st.empty()
            full_response = ""
            
            # Use DeepSeek for the high-speed "Gemini" feel
            with st.spinner(" "): 
                stream = stream_ai(MODELS["deepseek"], prompt or "Analyze files", ctx)
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        placeholder.markdown(full_response + "▌") # Animated cursor
            
            placeholder.code(full_response) if "this ai" not in full_response else placeholder.markdown(full_response)
            ans = full_response
            
        st.session_state.messages.append({"role": "assistant", "content": ans})
        # Log in background
        with open("usage_logs.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] {st.user.email}: {prompt[:30]}\n")
