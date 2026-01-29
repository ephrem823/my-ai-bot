import streamlit as st
import os
import datetime
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CORE CONFIG ---
MODELS = {
    "primary": "deepseek-ai/DeepSeek-V3",
    "fast_check": "zai-org/GLM-4.7-Flash"
}
ADMIN_EMAIL = "efaxalemayehu@gmail.com"

st.set_page_config(page_title="AMEK AI", layout="wide", page_icon="🪄")

# --- 2. GEMINI UI & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #131314; color: #E3E3E3; }
    [data-testid="stSidebar"] { background-color: #1E1F20 !important; border: none; }
    
    /* Skeleton Loader */
    @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    .skeleton { height: 18px; background: #3C4043; border-radius: 4px; margin-bottom: 8px; animation: pulse 1.5s infinite; }
    
    /* Gemini Input Bar (Standard st.chat_input with file support) */
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

# --- 4. SIDEBAR (LOGIC FOR ADMIN VS USER) ---
with st.sidebar:
    st.markdown("### 📂 AMEK Controller")
    
    if st.user.is_logged_in:
        st.write(f"Verified: **{st.user.name}**")
        if st.button("🚪 Logout"): st.logout()
        
        # Admin-only section (No "unlock" message for users)
        if st.user.email == ADMIN_EMAIL:
            st.divider()
            st.markdown("👑 **Admin Console**")
            if os.path.exists("usage_logs.txt"):
                with open("usage_logs.txt", "r") as f: st.text(f.read())
                if st.button("🗑️ Wipe System Logs", type="primary"):
                    os.remove("usage_logs.txt")
                    st.rerun()
    else:
        st.info("Log in after your first prompt to see AMEK's reasoning.")

# --- 5. CHAT INTERFACE ---
st.markdown("<h1 style='text-align: center;'>AMEK</h1>", unsafe_allow_html=True)

# Display Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "this ai" not in m["content"]: 
            st.code(m["content"])
        else: 
            st.markdown(m["content"])

# --- 6. GEMINI INPUT AREA (TEXT + FILE UPLOAD) ---
# accept_file="multiple" adds the "+" sign inside the input area like Gemini
chat_data = st.chat_input("Ask AMEK or attach files...", accept_file="multiple", file_type=["pdf", "py", "txt", "js"])

if chat_data:
    prompt = chat_data.text
    files = chat_data.files
    
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt or "Uploaded Files"})
    with st.chat_message("user"): 
        if prompt: st.markdown(prompt)
        if files: st.info(f"📎 {len(files)} file(s) attached for context.")

    # Check Login
    if not st.user.is_logged_in:
        with st.chat_message("assistant"):
            st.warning("🔐 Verification Required")
            st.markdown("Please sign in to process code and analyze documents.")
            st.button("Continue with Google", on_click=st.login)
    else:
        # AI Processing
        with st.chat_message("assistant"):
            clean_p = prompt.lower().strip() if prompt else ""
            
            if clean_p == "hey":
                res = "hi"
                st.markdown(res)
            else:
                # Skeleton Loader (Visual feedback)
                load_box = st.empty()
                load_box.markdown('<div class="skeleton"></div><div class="skeleton" style="width:70%"></div>', unsafe_allow_html=True)
                
                client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
                
                # Fast Rejection Check
                check_msg = [{"role": "system", "content": "If this is not coding/IT/Logic, say 'REJECT'. Otherwise 'OK'."}, {"role": "user", "content": prompt or "File analysis"}]
                check = client.chat.completions.create(model=MODELS["fast_check"], messages=check_msg, max_tokens=10).choices[0].message.content
                
                if "REJECT" in check.upper() and clean_p != "hey":
                    load_box.empty()
                    res = "this ai is only for code except hey"
                    st.markdown(res)
                else:
                    # RAG Context Building
                    ctx = ""
                    if files:
                        for f in files:
                            if f.type == "application/pdf":
                                with open("temp_rag.pdf", "wb") as tmp: tmp.write(f.getvalue())
                                ctx += "\n".join([d.page_content for d in PyPDFLoader("temp_rag.pdf").load()])
                            else:
                                ctx += f.read().decode("utf-8")

                    # --- 7. SAFE STREAMING LOOP ---
                    load_box.empty()
                    full_res = ""
                    place = st.empty()
                    
                    stream = client.chat.completions.create(
                        model=MODELS["primary"],
                        messages=[{"role": "system", "content": "Strict Code Assistant."}, {"role": "user", "content": f"Context: {ctx}\n\nTask: {prompt}"}],
                        stream=True, 
                        max_tokens=2500
                    )
                    
                    for chunk in stream:
                        # Safety check for choices list
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            token = chunk.choices[0].delta.content
                            if token:
                                full_res += token
                                place.markdown(full_res + "▌") # Animated cursor
                    
                    place.code(full_res)
                    res = full_res

            st.session_state.messages.append({"role": "assistant", "content": res})
            # Log usage
            with open("usage_logs.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] {st.user.email}: {prompt[:20]}...\n")
