# 🪄 AMEK AI - Professional Code Generator

A sophisticated AI-powered code generation and assistance platform built with Streamlit and Hugging Face models.

## ✨ Features

- **Advanced Code Generation**: Multi-language support with intelligent code completion
- **Secure Chat System**: User authentication with AWS integration
- **Persistent Storage**: SQLite database for chat history and user data
- **Rate Limiting**: Built-in protection against abuse
- **Cost Tracking**: Monitor token usage and API costs
- **Export Options**: Download chats in Markdown, JSON, or plain text
- **Share Conversations**: Generate secure links to share chats
- **Admin Dashboard**: Analytics and system monitoring
- **Responsive Design**: Professional dark theme with mobile support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Hugging Face account and API token
- AWS account (for authentication)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd my-ai-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Primary Hugging Face API token | Required |
| `HF_TOKEN_SECONDARY` | Backup API token | Optional |
| `ADMIN_EMAIL` | Admin user email | Required |
| `SECRET_KEY` | JWT secret key | Auto-generated |
| `MAX_REQUESTS_PER_MINUTE` | Rate limit per user | 20 |
| `MAX_TOKENS_PER_REQUEST` | Token limit per request | 2500 |
| `DATABASE_PATH` | SQLite database file | chats.db |
| `MONTHLY_BUDGET_USD` | Cost tracking budget | 100.0 |

### Supported Models

- **Primary**: `deepseek-ai/DeepSeek-V3` - Main code generation model
- **Fast Check**: `zai-org/GLM-4.7-Flash` - Quick responses and title generation

## 🏗️ Architecture

### Core Components

1. **Configuration Management** (`Config` class)
   - Centralized settings
   - Environment variable handling
   - Model configuration

2. **Security Layer**
   - Input sanitization
   - JWT token management
   - Rate limiting
   - Audit logging

3. **Database Layer** (`ChatDatabase` class)
   - User management
   - Conversation storage
   - Message tracking
   - Cost monitoring

4. **AI Integration** (`AIClientManager` class)
   - Hugging Face API management
   - Fallback handling
   - Response caching

5. **Analytics** (`MetricsCollector` class)
   - Performance monitoring
   - Usage statistics
   - Error tracking

### Database Schema

```sql
-- Users table
users (user_id, email, name, created_at, last_active, total_tokens_used, total_cost)

-- Conversations table  
conversations (conversation_id, user_id, title, created_at, updated_at, is_active, is_shared, share_token, message_count)

-- Messages table
messages (message_id, conversation_id, role, content, timestamp, tokens_used, model_used, processing_time)

-- Cost tracking table
cost_tracking (id, user_id, date, tokens_used, cost, model_used)

-- Audit logs table
audit_logs (id, timestamp, user_email, action, details, ip_address, success)
```

## 🔒 Security Features

- **Input Sanitization**: XSS and SQL injection prevention
- **Rate Limiting**: Per-user request throttling
- **Audit Logging**: Comprehensive activity tracking
- **Secure Tokens**: JWT-based sharing with expiration
- **Data Validation**: Input validation and error handling

## 📊 Analytics & Monitoring

### User Dashboard
- Token usage statistics
- Cost tracking
- Conversation metrics
- Activity history

### Admin Dashboard
- System-wide analytics
- User activity monitoring
- Performance metrics
- Error tracking

## 🔧 Development

### Project Structure
```
my-ai-bot/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env.example       # Environment template
├── README.md          # Documentation
├── chats.db           # SQLite database (auto-created)
└── chat_histories/    # Export directory (auto-created)
```

### Key Classes

- `Config`: Configuration management
- `SecurityManager`: Security operations
- `RateLimiter`: Request throttling
- `AuditLogger`: Activity logging
- `ChatDatabase`: Data persistence
- `MetricsCollector`: Analytics
- `ResponseCache`: Response caching
- `AIClientManager`: AI model management
- `ConversationExporter`: Export utilities

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Configure secrets in dashboard
4. Deploy

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Environment Variables for Production
```bash
# Required
HF_TOKEN=your_production_token
ADMIN_EMAIL=admin@yourcompany.com

# Optional but recommended
HF_TOKEN_SECONDARY=backup_token
SECRET_KEY=secure_random_key
MAX_REQUESTS_PER_MINUTE=50
MONTHLY_BUDGET_USD=500
```

## 📝 Usage Examples

### Basic Chat
1. Log in with AWS credentials
2. Click "New Chat" to start
3. Type your coding question
4. Get AI-generated response with code examples

### Code Generation
```
User: "Create a Python function to validate email addresses"
AMEK: [Provides complete function with validation logic, error handling, and usage examples]
```

### Export & Share
1. Select conversation from sidebar
2. Click export format (Markdown/JSON)
3. Or click "Share Chat" for public link

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- Check the logs in the admin dashboard
- Review audit logs for debugging
- Monitor token usage and costs
- Contact admin for access issues

---

**Built with ❤️ using Streamlit and Hugging Face**