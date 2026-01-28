# ⚡ AMEK_v1.0: Advanced Multi-model Ensemble Kernel

**AMEK** is a high-performance, code-centric AI assistant. It utilizes a **Triple-Brain Ensemble** architecture and **RAG (Retrieval-Augmented Generation)** to deliver precise, expert-level programming solutions while strictly filtering out non-technical queries.

![AMEK Interface](https://img.shields.io/badge/Interface-Cyberpunk_Dark-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

---

## 🚀 Core Technologies

### 1. Triple-Brain Ensemble
AMEK doesn't rely on just one model. It processes every coding request through three industry-leading models simultaneously:
* **Qwen 2.5 Coder** (Instruction Following)
* **DeepSeek V3** (Complex Logic & Merging)
* **GLM 4.7** (Speed & Flash Reasoning)

### 2. Knowledge Ingestion (RAG)
By uploading documentation or project files in PDF format, AMEK uses **FAISS** vector storage and **SentenceTransformers** to search your private data before generating code. This ensures the AI knows *your* specific project rules.



### 3. Strict Coding Filter
AMEK is built for work. It features a hard-coded rejection layer:
* **Input:** Coding/Technical → **Full Execution**
* **Input:** "Hi/Hey" → **Greeting**
* **Input:** General Talk → **Rejection:** *"this ai is only for code except hey"*

---

## 🛠️ Installation

1. **Clone the System:**
   ```bash
   git clone [https://github.com/ephrem823/amek-ai.git](https://github.com/ephrem823/amek-ai.git)
   cd amek-ai
   git clone [https://github.com/ephrem823/my-ai-bot.git](https://github.com/ephrem823/my-ai-bot.git)
   cd my-ai-bot
