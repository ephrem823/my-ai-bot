import streamlit as st
import os
import datetime
import json
import uuid
import hashlib
import secrets
import time
import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import defaultdict
import pandas as pd

# Third-party imports
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import bleach

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration management"""
    # AI Models
    MODELS = {
        "primary": "deepseek-ai/DeepSeek-V3",
        "fast_check": "zai-org/GLM-4.7-Flash"
    }
    
    # Security
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "efaxalemayehu@gmail.com")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_TOKEN_SECONDARY = os.getenv("HF_TOKEN_SECONDARY", "")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20"))
    MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2500"))
    
    # Storage
    DATABASE_PATH = os.getenv("DATABASE_PATH", "chats.db")
    CHAT_HISTORY_DIR = "chat_histories"
    
    # Features
    ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    ENABLE_FILE_UPLOAD = os.getenv("ENABLE_FILE_UPLOAD", "true").lower() == "true"
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    # Cost Management
    MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET_USD", "100.0"))
    TOKEN_COSTS = {
        "deepseek-ai/DeepSeek-V3": 0.00002,
        "zai-org/GLM-4.7-Flash": 0.000001
    }

config = Config()

# ============================================================================
# SECURITY & UTILITIES
# ============================================================================

class SecurityManager:
    """Handles security operations"""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Prevent XSS and injection attacks"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = bleach.clean(text, tags=[], strip=True)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r"('\s*(or|and)\s*')",
            r"(--)",
            r"(/\*|\*/)",
            r"(;\s*drop\s+table)",
            r"(;\s*delete\s+from)"
        ]
        for pattern in sql_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def generate_token(data: dict, expiry_days: int = 7) -> str:
        """Generate secure token for sharing"""
        import jwt
        payload = {
            **data,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=expiry_days)
        }
        return jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")
    
    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """Verify and decode token"""
        import jwt
        try:
            return jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        except:
            return None

class RateLimiter:
    """Rate limiting to prevent abuse"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_email: str) -> Tuple[bool, Optional[str]]:
        """Check if user is within rate limits"""
        now = datetime.datetime.now()
        window_start = now - datetime.timedelta(minutes=1)
        
        # Clean old requests
        self.requests[user_email] = [
            req_time for req_time in self.requests[user_email]
            if req_time > window_start
        ]
        
        current_count = len(self.requests[user_email])
        
        if current_count >= config.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - min(self.requests[user_email])).seconds
            return False, f"Rate limit exceeded. Please wait {wait_time} seconds."
        
        self.requests[user_email].append(now)
        return True, None

class AuditLogger:
    """Comprehensive audit logging"""
    
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()
    
    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_email TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                success BOOLEAN DEFAULT 1
            )
        """)
        self.conn.commit()
    
    def log(self, user_email: str, action: str, details: str = "", success: bool = True):
        """Log an action"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (user_email, action, details, success)
            VALUES (?, ?, ?, ?)
        """, (user_email, action, details, success))
        self.conn.commit()
    
    def get_user_activity(self, user_email: str, limit: int = 100) -> pd.DataFrame:
        """Get recent user activity"""
        query = """
            SELECT timestamp, action, details, success
            FROM audit_logs
            WHERE user_email = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(user_email, limit))

# ============================================================================
# DATABASE LAYER
# ============================================================================

class ChatDatabase:
    """Professional SQLite database for chat management"""
    
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._create_indexes()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_tokens_used INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                is_shared BOOLEAN DEFAULT 0,
                share_token TEXT,
                message_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER DEFAULT 0,
                model_used TEXT,
                processing_time REAL DEFAULT 0.0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        # Cost tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date DATE DEFAULT CURRENT_DATE,
                tokens_used INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                model_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create indexes for performance"""
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_share_token ON conversations(share_token)")
        self.conn.commit()
    
    def create_or_get_user(self, email: str, name: str = "") -> str:
        """Create user or return existing user_id"""
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        
        if result:
            # Update last active
            cursor.execute(
                "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE email = ?",
                (email,)
            )
            self.conn.commit()
            return result[0]
        
        # Create new user
        user_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO users (user_id, email, name) VALUES (?, ?, ?)",
            (user_id, email, name)
        )
        self.conn.commit()
        return user_id
    
    def create_conversation(self, user_id: str, title: str = None) -> str:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())[:8]
        if not title:
            title = f"Chat {datetime.datetime.now().strftime('%b %d, %H:%M')}"
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (conversation_id, user_id, title)
            VALUES (?, ?, ?)
        """, (conversation_id, user_id, title))
        self.conn.commit()
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str,
                   tokens_used: int = 0, model_used: str = "", processing_time: float = 0.0) -> str:
        """Add message to conversation"""
        message_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (message_id, conversation_id, role, content, tokens_used, model_used, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (message_id, conversation_id, role, content, tokens_used, model_used, processing_time))
        
        # Update conversation
        cursor.execute("""
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP,
                message_count = message_count + 1
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        self.conn.commit()
        return message_id
    
    def get_conversation_messages(self, conversation_id: str) -> List[dict]:
        """Get all messages in a conversation"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content, timestamp, tokens_used, model_used, processing_time
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "tokens_used": row[3],
                "model_used": row[4],
                "processing_time": row[5]
            })
        return messages
    
    def get_user_conversations(self, user_id: str) -> List[dict]:
        """Get all conversations for a user"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT conversation_id, title, created_at, updated_at, message_count, is_shared
            FROM conversations
            WHERE user_id = ? AND is_active = 1
            ORDER BY updated_at DESC
        """, (user_id,))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "conversation_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4],
                "is_shared": bool(row[5])
            })
        return conversations
    
    def delete_conversation(self, conversation_id: str):
        """Soft delete a conversation"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversations SET is_active = 0 WHERE conversation_id = ?",
            (conversation_id,)
        )
        self.conn.commit()
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (title, conversation_id)
        )
        self.conn.commit()
    
    def search_conversations(self, user_id: str, query: str) -> List[dict]:
        """Search conversations by content"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT c.conversation_id, c.title, c.updated_at
            FROM conversations c
            JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ? AND c.is_active = 1
            AND (m.content LIKE ? OR c.title LIKE ?)
            ORDER BY c.updated_at DESC
            LIMIT 20
        """, (user_id, f"%{query}%", f"%{query}%"))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "conversation_id": row[0],
                "title": row[1],
                "updated_at": row[2]
            })
        return results
    
    def create_share_token(self, conversation_id: str) -> str:
        """Generate share token for conversation"""
        token = secrets.token_urlsafe(16)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE conversations
            SET is_shared = 1, share_token = ?
            WHERE conversation_id = ?
        """, (token, conversation_id))
        self.conn.commit()
        return token
    
    def get_conversation_by_token(self, token: str) -> Optional[dict]:
        """Get conversation by share token"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT conversation_id, title, created_at
            FROM conversations
            WHERE share_token = ? AND is_shared = 1
        """, (token,))
        
        row = cursor.fetchone()
        if row:
            return {
                "conversation_id": row[0],
                "title": row[1],
                "created_at": row[2]
            }
        return None
    
    def track_cost(self, user_id: str, tokens_used: int, model_used: str):
        """Track token usage and cost"""
        cost = tokens_used * config.TOKEN_COSTS.get(model_used, 0)
        cursor = self.conn.cursor()
        
        # Update user totals
        cursor.execute("""
            UPDATE users
            SET total_tokens_used = total_tokens_used + ?,
                total_cost = total_cost + ?
            WHERE user_id = ?
        """, (tokens_used, cost, user_id))
        
        # Track daily cost
        cursor.execute("""
            INSERT INTO cost_tracking (user_id, tokens_used, cost, model_used)
            VALUES (?, ?, ?, ?)
        """, (user_id, tokens_used, cost, model_used))
        
        self.conn.commit()
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get user statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT total_tokens_used, total_cost,
                   (SELECT COUNT(*) FROM conversations WHERE user_id = ? AND is_active = 1) as conv_count,
                   (SELECT SUM(message_count) FROM conversations WHERE user_id = ? AND is_active = 1) as total_messages
            FROM users
            WHERE user_id = ?
        """, (user_id, user_id, user_id))
        
        row = cursor.fetchone()
        if row:
            return {
                "total_tokens": row[0] or 0,
                "total_cost": row[1] or 0.0,
                "conversation_count": row[2] or 0,
                "total_messages": row[3] or 0
            }
        return {}

# ============================================================================
# ANALYTICS & MONITORING
# ============================================================================

@dataclass
class Metric:
    timestamp: datetime.datetime
    metric_name: str
    value: float
    user_email: str

class MetricsCollector:
    """Collect and analyze metrics"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
    
    def track(self, user_email: str, metric_name: str, value: float):
        """Track a metric"""
        self.metrics.append(Metric(
            timestamp=datetime.datetime.now(),
            metric_name=metric_name,
            value=value,
            user_email=user_email
        ))
        
        # Keep only last 1000 metrics in memory
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_dashboard_data(self) -> dict:
        """Get analytics dashboard data"""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        
        # Calculate statistics
        response_times = df[df.metric_name == "response_time"]["value"]
        tokens = df[df.metric_name == "tokens_used"]["value"]
        errors = df[df.metric_name == "error"]
        
        return {
            "total_requests": len(df),
            "avg_response_time": response_times.mean() if len(response_times) > 0 else 0,
            "total_tokens": tokens.sum() if len(tokens) > 0 else 0,
            "error_count": len(errors),
            "error_rate": (len(errors) / len(df) * 100) if len(df) > 0 else 0,
            "active_users": df["user_email"].nunique(),
            "requests_last_hour": len(df[df.timestamp > datetime.datetime.now() - datetime.timedelta(hours=1)])
        }

# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """Cache responses for identical prompts"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, prompt: str, context: str = "") -> str:
        """Generate cache key"""
        content = f"{prompt}||{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, context: str = "") -> Optional[str]:
        """Get cached response"""
        key = self._get_key(prompt, context)
        entry = self.cache.get(key)
        
        if entry:
            # Check if cache is still fresh (24 hours)
            if datetime.datetime.now() - entry["timestamp"] < datetime.timedelta(hours=24):
                return entry["response"]
        return None
    
    def set(self, prompt: str, response: str, context: str = ""):
        """Cache a response"""
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        key = self._get_key(prompt, context)
        self.cache[key] = {
            "response": response,
            "timestamp": datetime.datetime.now()
        }

# ============================================================================
# AI CLIENT MANAGER
# ============================================================================

class AIClientManager:
    """Manage AI model connections"""
    
    def __init__(self):
        self.primary_client = InferenceClient(api_key=config.HF_TOKEN)
        self.backup_client = InferenceClient(api_key=config.HF_TOKEN_SECONDARY) if config.HF_TOKEN_SECONDARY else None
        self.use_backup = False
    
    def get_client(self) -> InferenceClient:
        """Get active client"""
        if self.use_backup and self.backup_client:
            return self.backup_client
        return self.primary_client
    
    def switch_to_backup(self):
        """Switch to backup token"""
        if self.backup_client:
            self.use_backup = True

# ============================================================================
# EXPORT UTILITIES
# ============================================================================

class ConversationExporter:
    """Export conversations in various formats"""
    
    @staticmethod
    def to_markdown(messages: List[dict], title: str = "Conversation") -> str:
        """Export to Markdown"""
        md = f"# {title}\n\n"
        md += f"*Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        md += "---\n\n"
        
        for msg in messages:
            role = "**You**" if msg["role"] == "user" else "**AMEK**"
            md += f"{role}:\n```\n{msg['content']}\n```\n\n"
        
        return md
    
    @staticmethod
    def to_json(messages: List[dict], title: str = "Conversation") -> str:
        """Export to JSON"""
        export_data = {
            "title": title,
            "exported_at": datetime.datetime.now().isoformat(),
            "messages": messages
        }
        return json.dumps(export_data, indent=2)
    
    @staticmethod
    def to_text(messages: List[dict]) -> str:
        """Export to plain text"""
        text = ""
        for msg in messages:
            role = "You" if msg["role"] == "user" else "AMEK"
            text += f"{role}: {msg['content']}\n\n"
        return text

# ============================================================================
# INITIALIZE GLOBAL INSTANCES
# ============================================================================

@st.cache_resource
def get_database():
    """Get database instance (cached)"""
    return ChatDatabase()

@st.cache_resource
def get_security_manager():
    """Get security manager (cached)"""
    return SecurityManager()

@st.cache_resource
def get_rate_limiter():
    """Get rate limiter (cached)"""
    return RateLimiter()

@st.cache_resource
def get_audit_logger():
    """Get audit logger (cached)"""
    return AuditLogger()

@st.cache_resource
def get_metrics_collector():
    """Get metrics collector (cached)"""
    return MetricsCollector()

@st.cache_resource
def get_response_cache():
    """Get response cache (cached)"""
    return ResponseCache()

@st.cache_resource
def get_ai_client_manager():
    """Get AI client manager (cached)"""
    return AIClientManager()

# Initialize
db = get_database()
security = get_security_manager()
rate_limiter = get_rate_limiter()
audit_logger = get_audit_logger()
metrics = get_metrics_collector()
response_cache = get_response_cache()
ai_manager = get_ai_client_manager()

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AMEK AI - Professional Code Generator",
    layout="wide",
    page_icon="🪄",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Base Theme */
    .stApp {
        background-color: #131314;
        color: #E3E3E3;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E1F20 !important;
        border: none;
    }
    
    /* Skeleton Loader */
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    
    .skeleton {
        height: 18px;
        background: #3C4043;
        border-radius: 4px;
        margin-bottom: 8px;
        animation: pulse 1.5s infinite;
    }
    
    /* Chat History */
    .chat-item {
        padding: 12px;
        margin: 6px 0;
        background-color: #2C2D2E;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        border-left: 3px solid transparent;
    }
    
    .chat-item:hover {
        background-color: #3C4043;
        border-left-color: #8AB4F8;
    }
    
    .chat-item-active {
        background-color: #3C4043;
        border-left-color: #8AB4F8;
    }
    
    /* Code Blocks with Copy Button */
    .code-container {
        position: relative;
        background: #1E1F20;
        border-left: 3px solid #8AB4F8;
        padding: 12px;
        border-radius: 8px;
        margin: 12px 0;
    }
    
    .copy-btn {
        position: absolute;
        top: 8px;
        right: 8px;
        background: #3C4043;
        color: #E3E3E3;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        transition: background 0.2s;
    }
    
    .copy-btn:hover {
        background: #5C6063;
    }
    
    /* Status Messages */
    .status-success {
        background: #1F3C2E;
        border-left: 4px solid #81C995;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .status-error {
        background: #3C1F1F;
        border-left: 4px solid #F28B82;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .status-warning {
        background: #3C3520;
        border-left: 4px solid #FDD663;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: #2C2D2E;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #3C4043;
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #8AB4F8;
    }
    
    .metric-label {
        font-size: 12px;
        color: #9AA0A6;
        margin-top: 4px;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-radius: 32px !important;
        background-color: #1E1F20 !important;
        border: 1px solid #3C4043 !important;
    }
    
    .stChatMessage {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .stApp { padding: 8px; }
        .chat-item { font-size: 13px; padding: 10px; }
        .metric-card { padding: 12px; }
    }
    
    /* Accessibility */
    button:focus, input:focus {
        outline: 2px solid #8AB4F8 !important;
        outline-offset: 2px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1E1F20;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3C4043;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5C6063;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ============================================================================
# SIDEBAR - CHAT HISTORY & CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("### 🪄 AMEK AI")
    st.caption("Professional Code Generator v2.0")
    
    if st.user.is_logged_in:
        # Get or create user
        if st.session_state.user_id is None:
            st.session_state.user_id = db.create_or_get_user(st.user.email, st.user.name)
            audit_logger.log(st.user.email, "login", "User logged in")
        
        # User Info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"✨ **{st.user.name}**")
        with col2:
            if st.button("🚪", help="Logout"):
                audit_logger.log(st.user.email, "logout", "User logged out")
                st.session_state.user_id = None
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.logout()
        
        st.divider()
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            conversation_id = db.create_conversation(st.session_state.user_id)
            st.session_state.current_chat_id = conversation_id
            st.session_state.messages = []
            audit_logger.log(st.user.email, "new_chat", f"Created chat {conversation_id}")
            st.rerun()
        
        # Search
        search_query = st.text_input("🔍 Search chats", key="search_input", placeholder="Search...")
        
        st.divider()
        st.markdown
