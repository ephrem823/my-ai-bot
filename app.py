import streamlit as st
import os
import datetime
import concurrent.futures
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CORE SETTINGS ---
# Using Fast variants for speed
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}
ADMIN_EMAIL = "efaxalemayehu@gmail.com"

st.set_page_config(page_title="AMEK AI", layout="wide", page_icon="🧠")

# --- 2. THEME & UI ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stChatMessage { border-radius: 12px; border: 1px solid #30363d; background-color: #161b22; margin-bottom: 10px; }
    .stChatInputContainer { border: 1px solid #58a6ff !important; border-radius: 8px; }
    h1 { color: #58a6ff !important; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGGING & ADMIN LOGIC ---
def log_usage(email, query, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {email}: {query[:40]}... -> {response[:20]}...\n"
    with open("usage_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

# --- 4. ENGINE SETUP (CACHED) ---
@st.cache_resource
def load_engine():
    return SentenceTransformer("all-MiniLM-L6-v2")

class AMEKEmbedder:
    def __init__(self, model): self.model = model
    def embed_documents(self, t): return self.model.encode(t).tolist()
    def embed_query(self, t): return self.model.encode([t])[0].tolist()

raw_model = load_engine()
embedder = AMEKEmbedder(raw_model)

if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None

# --- 5. SIDEBAR (LOGOUT & ADMIN) ---
with st.sidebar:
    st.title("🧠 AMEK CORE")
    if st.user.is_logged_in:
        st.write(f"Logged in: **{st.user.name}**")
        if st.button("🚪 Logout"): st.logout()
    else:
        st.warning("🔒 Features Locked")
    
    st.divider()
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.rerun()

    # Admin Control Panel
    if st.user.is_logged_in and st.user.email == ADMIN_EMAIL:
        st.markdown("### 👑 Admin Logs")
        if os.path.exists("usage_logs.txt"):
            with open("usage_logs.txt", "r") as f: st.text(f.read())
            if st.button("🗑️ WIPE ALL LOGS", type="primary"):
                os.remove("usage_logs.txt")
                st.rerun()
        else: st.info("Logs are clear.")

# --- 6. AI INTERFACE ---
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

def call_amek_ai(m_id, prompt, context="", stream=False):
    sys_inst = "You are AMEK. Strict Code Assistant. If input is not code-related, reply: 'this ai is only for code except hey'."
    full_p = f"Context: {context}\n\nTask: {prompt}" if context else prompt
    msg = [{"role": "system", "content": sys_inst}, {"role": "user", "content": full_p}]
    try:
        return client.chat.completions.create(model=m_id, messages=msg, stream=stream, max_tokens=1500)
    except: return "Connection Error. Please check HF_TOKEN."

# --- 7. MAIN CHAT DISPLAY ---
st.title("🚀 AMEK COMMAND CENTER")

# Show history immediately (The "Not Blacked Out" look)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "this ai is only for code" not in m["content"]:
            st.code(m["content"])
        else:
            st.markdown(m["content"])

# --- 8. SMART INPUT (LOGIN GATE) ---
if not st.user.is_logged_in:
    # Visible UI, but blocked input
    st.info("👋 Hello! Please verify your account to unlock the AMEK Code Engine.")
    st.button("🔑 Login with Google to Chat", on_click=st.login, use_container_width=True)
else:
    # Full Unified Input Bar (Text + File)
    chat_input_data = st.chat_input("Ask AMEK or drop files...", accept_file=True, file_type=["pdf", "py", "txt", "js"])

    if chat_input_data:
        prompt = chat_input_data.text
        files = chat_input_data.files
        
        # Handle "hey" logic
        if prompt and prompt.lower().strip() == "hey":
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "hey"})
            st.rerun()

        # Process Files
        f_context = ""
        if files:
            for f in files:
                if f.type == "application/pdf":
                    with open("temp.pdf", "wb") as tmp: tmp.write(f.getvalue())
                    f_context += "\n".join([d.page_content for d in PyPDFLoader("temp.pdf").load()])
                else:
                    f_context += f.read().decode("utf-8")

        st.session_state.messages.append({"role": "user", "content": prompt or "📎 Uploaded Files"})
        st.chat_message("user").markdown(prompt or "📎 *Files uploaded for analysis.*")

        # Streaming Assistant Response
        with st.chat_message("assistant"):
            with st.status("🔮 AMEK Brain Working...") as status:
                # 1. Parallel Generation
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    futures = [ex.submit(call_amek_ai, m, prompt, f_context) for m in [MODELS["qwen"], MODELS["glm"]]]
                    results = [f.result().choices[0].message.content for f in futures]
                
                # 2. Check for Rejection
                if any("this ai is only for code" in r.lower() for r in results):
                    ans = "this ai is only for code except hey"
                    st.markdown(ans)
                else:
                    # 3. Final Streaming Merge
                    merge_p = f"Create the final code solution from these inputs:\n{results}"
                    stream_obj = call_amek_ai(MODELS["deepseek"], merge_p, stream=True)
                    ans = st.write_stream(stream_obj)
                
                status.update(label="✅ Ready", state="complete")
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
            log_usage(st.user.email, prompt or "FILE", ans)
