import streamlit as st
from huggingface_hub import InferenceClient
import concurrent.futures
import os

# --- 1. RAG & ENSEMBLE IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 2. CONFIGURATION & MODELS ---
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "glm": "zai-org/GLM-4.7-Flash"
}

st.set_page_config(page_title="ephrem AI SuperBrain", layout="wide", page_icon="🧠")

# Custom UI Styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; min-width: 280px; }
    .stButton>button { width: 100%; text-align: left; border: none; background: transparent; color: #c9d1d9; }
    .stButton>button:hover { background: #21262d; color: #58a6ff; }
    .stApp { background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INITIALIZE MODELS & SESSION ---
@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embed_model = load_embed_model()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} 
if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "New Chat"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR (HISTORY & RAG) ---
with st.sidebar:
    st.title("📜 Chat History")
    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.all_chats[st.session_state.current_chat_title] = st.session_state.messages
        st.session_state.messages = []
        st.session_state.current_chat_title = f"Chat {len(st.session_state.all_chats) + 1}"
        st.rerun()
    
    # List History
    for title in list(st.session_state.all_chats.keys()):
        if st.button(f"💬 {title[:20]}..."):
            st.session_state.current_chat_title = title
            st.session_state.messages = st.session_state.all_chats[title]
            st.rerun()

    st.divider()
    
    # RAG Section
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF to teach AI", type=["pdf"])
    if uploaded_file and st.button("🧠 Teach the AI"):
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embed_model)
            st.success("Knowledge Ingested!")

# --- 5. MAIN LOGIC ---
st.title(f"🧠 {st.session_state.current_chat_title}")

try:
    client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
except:
    st.error("Missing HF_TOKEN in Secrets!")
    st.stop()

def call_ai(model_id, prompt_text, context=""):
    system_instr = "You are a code generator. Provide ONLY code and bulleted descriptions. No small talk. Use the provided context if relevant."
    full_prompt = f"Context: {context}\n\nQuestion: {prompt_text}" if context else prompt_text
    try:
        messages = [{"role": "system", "content": system_instr}, {"role": "user", "content": full_prompt}]
        resp = client.chat.completions.create(model=model_id, messages=messages, max_tokens=1000)
        return resp.choices[0].message.content
    except: return ""

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="markdown")
        else:
            st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Hi or ask for code..."):
    if not st.session_state.messages:
        st.session_state.current_chat_title = prompt[:25]
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Check for Greeting
        if prompt.lower().strip() in ["hi", "hello", "hey"]:
            final_answer = "hey"
            st.markdown(final_answer)
        else:
            # 2. Get RAG Context
            context = ""
            if st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])

            # 3. Triple-Brain Ensemble
            with st.status("🧠 Consulting Triple-Brain Ensemble...") as status:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(call_ai, id, prompt, context) for id in MODELS.values()]
                    results = [f.result() for f in futures]
                
                status.update(label="✍️ Merging Best Answers...", state="running")
                merge_query = f"Merge these 3 solutions into one perfect code block + ### Logic Breakdown:\n1:{results[0]}\n2:{results[1]}\n3:{results[2]}"
                final_answer = call_ai(MODELS["deepseek"], merge_query)
                status.update(label="✅ Ready", state="complete")
            
            st.code(final_answer, language="markdown")
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.all_chats[st.session_state.current_chat_title] = st.session_state.messages
