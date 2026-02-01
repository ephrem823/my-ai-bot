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
import bleach

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Google Authentication
from google_auth import init_google_auth, login_button, logout, is_logged_in, get_user

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

# Initialize Google Auth
init_google_auth()

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
    
    if is_logged_in():
        user = get_user()
        # Get or create user
        if st.session_state.user_id is None:
            st.session_state.user_id = db.create_or_get_user(user['email'], user['name'])
            audit_logger.log(user['email'], "login", "User logged in")
        
        # User Info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"✨ **{user['name']}**")
        with col2:
            if st.button("🚪", help="Logout"):
                audit_logger.log(user['email'], "logout", "User logged out")
                st.session_state.user_id = None
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                logout()
        
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
        
        # Chat History
        conversations = db.get_user_conversations(st.session_state.user_id)
        
        if search_query:
            conversations = db.search_conversations(st.session_state.user_id, search_query)
        
        if conversations:
            st.markdown("**Recent Chats**")
            for conv in conversations[:10]:  # Show last 10
                is_active = conv["conversation_id"] == st.session_state.current_chat_id
                
                # Chat item with custom styling
                chat_class = "chat-item-active" if is_active else "chat-item"
                
                if st.button(
                    f"💬 {conv['title'][:30]}{'...' if len(conv['title']) > 30 else ''}",
                    key=f"chat_{conv['conversation_id']}",
                    use_container_width=True,
                    type="secondary" if is_active else "primary"
                ):
                    st.session_state.current_chat_id = conv["conversation_id"]
                    st.session_state.messages = db.get_conversation_messages(conv["conversation_id"])
                    audit_logger.log(st.user.email, "load_chat", f"Loaded chat {conv['conversation_id']}")
                    st.rerun()
                
                # Show message count and date
                st.caption(f"📊 {conv['message_count']} msgs • {conv['updated_at'][:10]}")
        
        st.divider()
        
        # User Stats
        stats = db.get_user_stats(st.session_state.user_id)
        if stats:
            st.markdown("**📊 Your Stats**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chats", stats.get("conversation_count", 0))
                st.metric("Tokens", f"{stats.get('total_tokens', 0):,}")
            with col2:
                st.metric("Messages", stats.get("total_messages", 0))
                st.metric("Cost", f"${stats.get('total_cost', 0):.3f}")
        
        st.divider()
        
        # Export Options
        if st.session_state.current_chat_id and st.session_state.messages:
            st.markdown("**📤 Export Chat**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Markdown", use_container_width=True):
                    md_content = ConversationExporter.to_markdown(
                        st.session_state.messages,
                        f"Chat {st.session_state.current_chat_id}"
                    )
                    st.download_button(
                        "Download MD",
                        md_content,
                        f"chat_{st.session_state.current_chat_id}.md",
                        "text/markdown"
                    )
            
            with col2:
                if st.button("📋 JSON", use_container_width=True):
                    json_content = ConversationExporter.to_json(
                        st.session_state.messages,
                        f"Chat {st.session_state.current_chat_id}"
                    )
                    st.download_button(
                        "Download JSON",
                        json_content,
                        f"chat_{st.session_state.current_chat_id}.json",
                        "application/json"
                    )
            
            # Share Chat
            if st.button("🔗 Share Chat", use_container_width=True):
                share_token = db.create_share_token(st.session_state.current_chat_id)
                share_url = f"{st.get_option('server.baseUrlPath')}/shared/{share_token}"
                st.success(f"Share URL: {share_url}")
                audit_logger.log(st.user.email, "share_chat", f"Shared chat {st.session_state.current_chat_id}")
        
        # Admin Panel (if admin)
        if user['email'] == config.ADMIN_EMAIL:
            st.divider()
            st.markdown("**🔧 Admin Panel**")
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.show_admin = True
                st.rerun()
    
    else:
        # Login prompt
        st.markdown("### 🔐 Login Required")
        st.info("Please log in to use AMEK AI")
        login_button()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

def generate_ai_response(prompt: str, context: str = "", model: str = None) -> Tuple[str, int, float]:
    """Generate AI response with error handling and metrics"""
    start_time = time.time()
    
    if not model:
        model = config.MODELS["primary"]
    
    # Check cache first
    cached_response = response_cache.get(prompt, context)
    if cached_response:
        processing_time = time.time() - start_time
        return cached_response, 0, processing_time  # 0 tokens for cached
    
    try:
        client = ai_manager.get_client()
        
        # Prepare messages
        messages = [
            {"role": "system", "content": """You are AMEK, a professional AI code generator and assistant. 
            You provide high-quality, secure, and well-documented code solutions.
            Always explain your code and include best practices.
            Format code blocks with proper syntax highlighting."""}
        ]
        
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=config.MAX_TOKENS_PER_REQUEST,
            temperature=0.7,
            stream=False
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else len(content.split()) * 1.3
        
        processing_time = time.time() - start_time
        
        # Cache the response
        response_cache.set(prompt, content, context)
        
        return content, int(tokens_used), processing_time
        
    except Exception as e:
        # Try backup client
        if not ai_manager.use_backup and ai_manager.backup_client:
            ai_manager.switch_to_backup()
            return generate_ai_response(prompt, context, model)
        
        processing_time = time.time() - start_time
        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\n\nError: {str(e)}"
        return error_msg, 0, processing_time

def display_message(message: dict, is_user: bool = False):
    """Display a chat message with proper formatting"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.markdown(message["content"])
        else:
            # Display AI response with code highlighting
            content = message["content"]
            
            # Check if content contains code blocks
            if "```" in content:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Regular text
                        if part.strip():
                            st.markdown(part)
                    else:  # Code block
                        lines = part.split('\n')
                        language = lines[0] if lines[0] else "text"
                        code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                        
                        if code.strip():
                            st.code(code, language=language)
            else:
                st.markdown(content)
            
            # Show metadata
            if message.get("tokens_used") or message.get("processing_time"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if message.get("model_used"):
                        st.caption(f"🤖 {message['model_used']}")
                with col2:
                    if message.get("tokens_used"):
                        st.caption(f"🎯 {message['tokens_used']} tokens")
                with col3:
                    if message.get("processing_time"):
                        st.caption(f"⚡ {message['processing_time']:.1f}s")

# Main chat interface
if is_logged_in():
    user = get_user()
    # Check rate limiting
    allowed, error_msg = rate_limiter.is_allowed(user['email'])
    if not allowed:
        st.error(error_msg)
        st.stop()
    
    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = db.create_conversation(st.session_state.user_id)
        audit_logger.log(st.user.email, "auto_new_chat", "Auto-created first chat")
    
    # Display chat title
    if st.session_state.current_chat_id:
        conversations = db.get_user_conversations(st.session_state.user_id)
        current_conv = next((c for c in conversations if c["conversation_id"] == st.session_state.current_chat_id), None)
        if current_conv:
            st.markdown(f"### 💬 {current_conv['title']}")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message, message["role"] == "user")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about code, development, or technology..."):
        # Sanitize input
        prompt = security.sanitize_input(prompt)
        
        if not prompt.strip():
            st.error("Please enter a valid message.")
            st.stop()
        
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Save to database
        db.add_message(
            st.session_state.current_chat_id,
            "user",
            prompt
        )
        
        # Display user message
        display_message(user_message, True)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Build context from recent messages
                context = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-5:]  # Last 5 messages
                    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages[:-1]])
                
                response, tokens_used, processing_time = generate_ai_response(prompt, context)
                
                # Track metrics
                metrics.track(user['email'], "response_time", processing_time)
                metrics.track(user['email'], "tokens_used", tokens_used)
                
                # Track costs
                if tokens_used > 0:
                    db.track_cost(st.session_state.user_id, tokens_used, config.MODELS["primary"])
                
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tokens_used": tokens_used,
                    "model_used": config.MODELS["primary"],
                    "processing_time": processing_time
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Save to database
                db.add_message(
                    st.session_state.current_chat_id,
                    "assistant",
                    response,
                    tokens_used,
                    config.MODELS["primary"],
                    processing_time
                )
                
                # Display response
                display_message(assistant_message)
                
                # Auto-generate title for first exchange
                if len(st.session_state.messages) == 2:
                    try:
                        title_prompt = f"Generate a short, descriptive title (max 50 chars) for this conversation: {prompt[:100]}"
                        title_response, _, _ = generate_ai_response(title_prompt, model=config.MODELS["fast_check"])
                        title = title_response.strip().strip('"').strip("'")[:50]
                        db.update_conversation_title(st.session_state.current_chat_id, title)
                    except:
                        pass  # Ignore title generation errors
                
                audit_logger.log(user['email'], "chat_message", f"Sent message in chat {st.session_state.current_chat_id}")
                st.rerun()

else:
    # Welcome screen for non-logged users
    st.markdown("""
    # 🪄 Welcome to AMEK AI
    
    ### Professional Code Generator & AI Assistant
    
    **Features:**
    - 🚀 Advanced code generation
    - 💡 Intelligent problem solving  
    - 📚 Multi-language support
    - 🔒 Secure & private conversations
    - 📊 Usage analytics
    - 💾 Chat history & export
    - 🔗 Share conversations
    
    **Supported Languages:**
    Python, JavaScript, TypeScript, Java, C++, Go, Rust, PHP, and more!
    
    ---
    
    ### 🔐 Get Started
    Please log in with your AWS account to begin using AMEK AI.
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_button()

# ============================================================================
# ADMIN DASHBOARD
# ============================================================================

if is_logged_in() and get_user()['email'] == config.ADMIN_EMAIL and st.session_state.get("show_admin"):
    st.markdown("---")
    st.markdown("## 🔧 Admin Dashboard")
    
    # Metrics overview
    dashboard_data = metrics.get_dashboard_data()
    
    if dashboard_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            """.format(dashboard_data.get("total_requests", 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            """.format(dashboard_data.get("avg_response_time", 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Tokens</div>
            </div>
            """.format(int(dashboard_data.get("total_tokens", 0))), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Active Users</div>
            </div>
            """.format(dashboard_data.get("active_users", 0)), unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    activity_df = audit_logger.get_user_activity("", 50)  # Get all users
    if not activity_df.empty:
        st.dataframe(activity_df, use_container_width=True)
    
    # System controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear Cache"):
            response_cache.cache.clear()
            st.success("Cache cleared!")
    
    with col2:
        if st.button("📊 Export Logs"):
            # Export audit logs
            st.download_button(
                "Download Logs",
                activity_df.to_csv(index=False),
                "audit_logs.csv",
                "text/csv"
            )
    
    if st.button("❌ Close Admin Panel"):
        st.session_state.show_admin = False
        st.rerun()

# ============================================================================
# SHARED CHAT VIEWER
# ============================================================================

# Handle shared chat URLs
query_params = st.query_params
if "shared" in query_params:
    token = query_params["shared"]
    shared_conv = db.get_conversation_by_token(token)
    
    if shared_conv:
        st.markdown(f"## 🔗 Shared Chat: {shared_conv['title']}")
        st.caption(f"Created: {shared_conv['created_at']}")
        
        # Load and display messages
        messages = db.get_conversation_messages(shared_conv["conversation_id"])
        for message in messages:
            display_message(message, message["role"] == "user")
        
        st.info("This is a read-only shared conversation. Log in to start your own chat!")
    else:
        st.error("Invalid or expired share link.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9AA0A6; font-size: 12px; padding: 20px;">
    🪄 AMEK AI v2.0 | Professional Code Generator<br>
    Built with ❤️ using Streamlit | Powered by Hugging Face
</div>
""", unsafe_allow_html=True)

# ============================================================================
# ERROR HANDLING & CLEANUP
# ============================================================================

# Global error handler
def handle_error(error):
    """Global error handler"""
    st.error(f"An unexpected error occurred: {str(error)}")
    if is_logged_in():
        user = get_user()
        audit_logger.log(user['email'], "error", str(error), success=False)
        metrics.track(user['email'], "error", 1)

# Set up error handling
import sys
sys.excepthook = lambda exc_type, exc_value, exc_traceback: handle_error(exc_value)

# Cleanup on app shutdown
import atexit
def cleanup():
    """Cleanup resources on shutdown"""
    try:
        db.conn.close()
        audit_logger.conn.close()
    except:
        pass

atexit.register(cleanup)