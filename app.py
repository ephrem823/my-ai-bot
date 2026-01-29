import streamlit as st
import os
import datetime
import concurrent.futures
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. MANDATORY LOGIN GATE ---
if not st.user.is_logged_in:
    st.title("🛡️ AMEK SECURITY")
    st.write("Access Restricted. Please identify yourself.")
    st.button("Log in with Google", on_click=st.login)
    st.stop()

# --- 2. DEFINE ADMIN & LOGGING ---
ADMIN_EMAIL = "efaxalemayehu@gmail.com"  # Change to your specific email
is_admin = (st.user.email == ADMIN_EMAIL)

def log_usage(query, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {st.user.email}: {query[:50]} | {response[:30]}\n"
    with open("usage_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

# --- 3. CONFIGURATION & MODELS ---
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}

st.set_page_config(page_title="AMEK", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stChatMessage { border-radius: 15px; border: 1px solid #30363d; background-color: #161b22; }
    .stop-btn > div > button { background-color: #da3633 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. ENGINE SETUP (CACHED) ---
@st.cache_resource
def load_embed(): return SentenceTransformer("all-MiniLM-L6-v2")

class EmbedWrapper:
    def __init__(self, model): self.model = model
    def embed_documents(self, t): return self.model.encode(t).tolist()
    def embed_query(self, t): return self.model.encode([t])[0].tolist()

embed_model = EmbedWrapper(load_embed())

if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.markdown(f"👤 **{st.user.name}**")
    if st.button("🚪 Logout"): st.logout()
    
    st.divider()
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("🛑 STOP ENGINE"): st.stop()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("➕ New Session"):
        st.session_state.messages = []
        st.rerun()

    if is_admin:
        st.divider()
        st.markdown("### 👑 Admin Logs")
        if os.path.exists("usage_logs.txt"):
            with open("usage_logs.txt", "r") as f: st.text(f.read())
            if st.button("🗑️ WIPE LOGS", type="primary"):
                os.remove("usage_logs.txt")
                st.rerun()
        else: st.info("No logs.")

# --- 6. AI CORE ---
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

def call_ai(m_id, prompt, context="", stream=False):
    sys = "Strict Code Assistant. If not code, say 'this ai is only for code except hey'."
    full = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": full}]
    try:
        if stream: return client.chat.completions.create(model=m_id, messages=messages, stream=True)
        return client.chat.completions.create(model=m_id, messages=messages).choices[0].message.content
    except: return "Connection Error"

# --- 7. CHAT & UPLOAD ---
st.title("🚀 AMEK COMMAND CENTER")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "this ai" not in m["content"]: st.code(m["content"])
        else: st.markdown(m["content"])

chat_data = st.chat_input("Ask or drop files...", accept_file=True, file_type=["pdf", "py", "txt"])

if chat_data:
    prompt, files = chat_input_data.text, chat_input_data.files
    f_context = ""
    if files:
        for f in files:
            if f.type == "application/pdf":
                with open("t.pdf", "wb") as tp: tp.write(f.getvalue())
                f_context += "\n".join([d.page_content for d in PyPDFLoader("t.pdf").load()])
            else: f_context += f.read().decode("utf-8")

    st.session_state.messages.append({"role": "user", "content": prompt or "[Files]"})
    st.chat_message("user").markdown(prompt or "📎 Attached files.")

    with st.chat_message("assistant"):
        if prompt and prompt.lower().strip() in ["hi", "hello", "hey"]:
            ans = "hey"
            st.markdown(ans)
        else:
            with st.status("🔮 AMEK Processing...") as status:
                # Parallel Check
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    futures = [ex.submit(call_ai, m, prompt, f_context) for m in [MODELS["qwen"], MODELS["glm"]]]
                    raw = [f.result() for f in futures]
                
                if any("this ai is only for code" in r.lower() for r in raw):
                    ans = "this ai is only for code except hey"
                    st.markdown(ans)
                else:
                    merge_p = f"Merge into 1 perfect code response:\n{raw}"
                    stream_data = call_ai(MODELS["deepseek"], merge_p, stream=True)
                    ans = st.write_stream(stream_data)
                status.update(label="✅ Ready", state="complete")
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        log_usage(prompt or "FILE", ans)
