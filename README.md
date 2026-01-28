# 🧠 ephrem AI SuperBrain (RAG Edition)

A high-performance AI assistant that uses a **Triple-Brain Ensemble** (Qwen, DeepSeek, GLM) and **RAG (Retrieval-Augmented Generation)** to provide expert-level coding advice and document analysis.

## 🚀 Key Features

* **Triple-Brain Ensemble:** Simultaneously queries three world-class models (`Qwen-2.5-Coder`, `DeepSeek-V3`, and `GLM-4`) and merges their outputs for the most accurate solution.
* **Knowledge Base (RAG):** Upload any PDF, and the AI will "learn" its contents instantly to provide context-aware answers.
* **Smart History:** Automatically saves and organizes your chat sessions in the sidebar.
* **Python 3.13 Ready:** Custom embedding wrapper to ensure high performance on the latest Python environments.
* **Clean UI:** Minimalist Dark Mode interface built with Streamlit.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM API:** [Hugging Face Inference API](https://huggingface.co/inference-api)
- **RAG Framework:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ephrem823/my-ai-bot.git](https://github.com/ephrem823/my-ai-bot.git)
   cd my-ai-bot
