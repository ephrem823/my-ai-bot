import streamlit as st
from huggingface_hub import InferenceClient
import concurrent.futures
import os
import datetime

# --- 1. RAG & UTILITY IMPORTS ---
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 2. CONFIGURATION & MODELS ---
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}

# --- 3. CUSTOM FRONTEND (CSS) ---
st.set_page_config(page_title="AMEK", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; text-shadow: 0px 0px 10px rgba(88, 166, 255, 0.3); }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; transition: 0.3s; }
    .stButton>button:hover { border-color: #58a6ff; color: #58a6ff; }
    /* Red Stop Button Styling */
    .stop-btn > div > button {
        background-color: #da3633 !important;
        color: white !important;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. THE EMBEDDING ENGINE ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

class EmbedderWrapper:
    def __init__(self, model): self.model = model
    def embed_documents(self, texts): return self.model.encode(texts).tolist()
    def embed_query(self, text): return self.model.encode([text])[0].tolist()

raw_model = load_embed_model()
embed_model = EmbedderWrapper(raw_model)

# --- 5. LOGGING SYSTEM ---
def log_usage(user_query, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] QUERY: {user_query} | AI: {response[:50]}...\n"
    with open("usage_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

# --- 6. STATE MANAGEMENT ---
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "all_chats" not in st.session_state: st.session_state.all_chats = {} 
if "current_chat_title" not in st.session_state: st.session_state.current_chat_title = "New Chat"
if "messages" not in st.session_state: st.session_state.messages = []

# --- 7. SIDEBAR UI ---
with st.sidebar:
    st.markdown("## 🧠 AMEK CONTROL")
    
    if st.button("➕ Start New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats[st.session_state.current_chat_title] = st.session_state.messages
        st.session_state.messages, st.session_state.current_chat_title = [], f"Chat {len(st.session_state.all_chats) + 1}"
        st.rerun()

    # STOP BUTTON
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("🛑 STOP GENERATION"):
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 📂 Permanent Knowledge")
    uploaded_pdf = st.file_uploader("Upload PDF to teach AI", type=["pdf"])
    if uploaded_pdf and st.button("🚀 Teach AI"):
        with st.spinner("Analyzing PDF..."):
            with open("temp.pdf", "wb") as f: f.write(uploaded_pdf.getvalue())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            st.session_state.vectorstore = FAISS.from_documents(chunks, embed_model)
            st.success("Knowledge Ingested!")

    # ADMIN LOGS VISUALIZER
    st.divider()
    with st.expander("🔐 Admin Usage Logs"):
        if os.path.exists("usage_logs.txt"):
            with open("usage_logs.txt", "r") as f:
                st.text(f.read())
        else:
            st.write("No activity recorded.")

# --- 8. AI CORE LOGIC ---
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

def call_ai(model_id, prompt_text, context=""):
    system_instr = """Strict Code Assistant. 
    If query is NOT code/programming, reply: 'this ai is only for code except hey'. 
    No chat, no greetings, just code and logic."""
    full_prompt = f"Context: {context}\n\nQuestion: {prompt_text}" if context else prompt_text
    try:
        msg = [{"role": "system", "content": system_instr}, {"role": "user", "content": full_prompt}]
        resp = client.chat.completions.create(model=model_id, messages=msg, max_tokens=1000)
        return resp.choices[0].message.content
    except: return "Connection Error"

# --- 9. CHAT INTERFACE ---
st.title(f"🚀 AMEK: {st.session_state.current_chat_title}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "this ai is only for code" in msg["content"] or msg["content"] == "hey":
            st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.code(msg["content"], language="markdown")
        else:
            st.markdown(msg["content"])

# --- 10. UNIFIED INPUT (Text + File) ---
with st.container():
    # File uploader right above the chat input
    chat_file = st.file_uploader("📎 Attach code/PDF for this prompt", type=["pdf", "txt", "py"], label_visibility="collapsed")
    
    if prompt := st.chat_input("Input your code query..."):
        
        # Immediate File Reading
        file_context = ""
        if chat_file:
            if chat_file.type == "application/pdf":
                with open("chat_temp.pdf", "wb") as f: f.write(chat_file.getvalue())
                loader = PyPDFLoader("chat_temp.pdf")
                file_context = "\n".join([d.page_content for d in loader.load()])
            else:
                file_context = chat_file.read().decode("utf-8")

        if not st.session_state.messages: st.session_state.current_chat_title = prompt[:25]
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            if prompt.lower().strip() in ["hi", "hello", "hey"]:
                final_answer = "hey"
                st.markdown(final_answer)
            else:
                # Build Context from File + Vectorstore
                combined_context = file_context
                if st.session_state.vectorstore:
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                    combined_context += "\n" + "\n".join([d.page_content for d in docs])

                with st.status("🔮 Consulting AI Ensemble...") as status:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(call_ai, id, prompt, combined_context) for id in MODELS.values()]
                        results = [f.result() for f in futures]
                    
                    if any("this ai is only for code" in r.lower() for r in results):
                        final_answer = "this ai is only for code except hey"
                    else:
                        merge_q = f"Merge into 1 perfect code block + logic. If not code, say 'this ai is only for code except hey':\n{results}"
                        final_answer = call_ai(MODELS["deepseek"], merge_q)
                    status.update(label="✅ Ready", state="complete")

                if "this ai is only for code" in final_answer:
                    st.markdown(final_answer)
                else:
                    st.code(final_answer, language="markdown")
            
            # Save to history and Log usage
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            log_usage(prompt, final_answer)
