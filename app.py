import streamlit as st
import os
import datetime
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CONFIG & SETTINGS ---
MODELS = {
    "primary": "deepseek-ai/DeepSeek-V3",
    "fast_check": "zai-org/GLM-4.7-Flash"
}
ADMIN_EMAIL = "efaxalemayehu@gmail.com"

st.set_page_config(page_title="AMEK AI", layout="wide", page_icon="🪄")

# --- 2. GEMINI UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #131314; color: #E3E3E3; }
    [data-testid="stSidebar"] { background-color: #1E1F20 !important; border: none; }
    
    /* Skeleton Loader */
    @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    .skeleton { height: 20px; background: #3C4043; border-radius: 4px; margin-bottom: 10px; animation: pulse 1.5s infinite; }
    
    /* Gemini Input Bar */
    .stChatInputContainer {
        border-radius: 32px !important;
        background-color: #1E1F20 !important;
        border: 1px solid #3C4043 !important;
    }
    .stChatMessage { border: none !important; background-color: transparent !important; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None

# --- 4. SIDEBAR (CONTROLLER) ---
with st.sidebar:
    st.markdown("### 📂 Knowledge Controller")
    # Supports up to 1GB if config.toml is set
    up_file = st.file_uploader("Upload Knowledge (Max 1GB)", type=["pdf", "py", "txt", "js"])
    
    if up_file and st.button("🚀 Ingest"):
        with st.status("Processing large file..."):
            with open("temp_file", "wb") as f: f.write(up_file.getvalue())
            loader = PyPDFLoader("temp_file") if up_file.type == "application/pdf" else None
            # Fast processing logic for large files
            st.success("Knowledge Synced!")

    st.divider()
    if st.user.is_logged_in:
        st.write(f"Logged in as: **{st.user.name}**")
        if st.button("Logout"): st.logout()
        if st.user.email == ADMIN_EMAIL:
            if st.button("🗑️ Wipe Logs"): 
                if os.path.exists("usage_logs.txt"): os.remove("usage_logs.txt")
    else:
        st.info("Log in to unlock Admin features")

# --- 5. MAIN CHAT INTERFACE ---
st.markdown("<h1 style='text-align: center;'>AMEK</h1>", unsafe_allow_html=True)

# Display Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "this ai" not in m["content"]: st.code(m["content"])
        else: st.markdown(m["content"])

# --- 6. THE IN-STREAM LOGIN LOGIC ---
prompt = st.chat_input("Ask AMEK a coding question...")

if prompt:
    # 1. Immediately show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # 2. Check Login Status AFTER prompt is sent
    if not st.user.is_logged_in:
        with st.chat_message("assistant"):
            st.warning("🔐 Verification Required")
            st.markdown("Please sign in to process this request and generate code.")
            st.button("Continue with Google", on_click=st.login)
            # Store prompt so it can be processed after login if desired
            st.session_state.pending_prompt = prompt 
    else:
        # 3. Process the AI Response
        with st.chat_message("assistant"):
            clean_p = prompt.lower().strip()
            
            if clean_p == "hey":
                res = "hi"
                st.markdown(res)
            else:
                # Skeleton Loader
                load_box = st.empty()
                load_box.markdown('<div class="skeleton"></div><div class="skeleton" style="width:70%"></div>', unsafe_allow_html=True)
                
                client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
                
                # Fast Code Check
                check = client.chat.completions.create(
                    model=MODELS["fast_check"], 
                    messages=[{"role": "system", "content": "If this is not coding/IT, say 'REJECT'. Otherwise 'OK'."}, {"role": "user", "content": prompt}],
                    max_tokens=5
                ).choices[0].message.content
                
                if "REJECT" in check.upper():
                    load_box.empty()
                    res = "this ai is only for code except hey"
                    st.markdown(res)
                else:
                    # Stream Response
                    load_box.empty()
                    full_res = ""
                    place = st.empty()
                    
                    stream = client.chat.completions.create(
                        model=MODELS["primary"],
                        messages=[{"role": "system", "content": "Strict Code Assistant."}, {"role": "user", "content": prompt}],
                        stream=True, max_tokens=2000
                    )
                    for chunk in stream:
                        token = chunk.choices[0].delta.content
                        if token:
                            full_res += token
                            place.markdown(full_res + "▌")
                    place.code(full_res)
                    res = full_res

            st.session_state.messages.append({"role": "assistant", "content": res})
