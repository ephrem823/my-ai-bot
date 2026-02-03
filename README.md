# 🪄 AMEK AI - Professional Code Generator

A powerful AI-powered code generation and assistance tool built with Streamlit and Hugging Face models.

## ✨ Features

- **Professional AI Code Generation** using DeepSeek-V3 and GLM-4.7-Flash models
- **Secure Input Sanitization** with XSS and SQL injection protection
- **Dual Token Support** with automatic fallback
- **Real-time Chat Interface** with syntax highlighting
- **Cost Tracking** and usage analytics
- **Professional Dark Theme** UI

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd "AI Bot/my-ai-bot"
python setup.py
```

### 2. Configure Environment
Edit `.env` file and add your Hugging Face API token:
```env
HF_TOKEN=hf_your_actual_token_here
HF_TOKEN_SECONDARY=hf_your_backup_token_here  # Optional
```

### 3. Run the Application
```bash
# Simple version
streamlit run app.py

# Full-featured version
streamlit run app_minimal.py
```

## 🔧 Configuration

### Required Environment Variables
- `HF_TOKEN`: Your primary Hugging Face API token
- `HF_TOKEN_SECONDARY`: Backup token (optional)

### Optional Configuration
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting (default: 20)
- `MAX_TOKENS_PER_REQUEST`: Token limit per request (default: 2500)
- `MONTHLY_BUDGET_USD`: Cost tracking budget (default: 100.0)

## 📁 Project Structure

```
my-ai-bot/
├── app.py              # Simple version
├── app_minimal.py      # Full-featured version
├── setup.py           # Setup script
├── requirements.txt   # Dependencies
├── .env              # Environment variables
├── .env.example      # Environment template
└── README.md         # This file
```

## 🛠️ Dependencies

- `streamlit>=1.28.0` - Web interface
- `huggingface_hub>=0.19.0` - AI model access
- `python-dotenv>=1.0.0` - Environment management
- `bleach>=6.0.0` - Input sanitization

## 🔒 Security Features

- Input sanitization against XSS attacks
- SQL injection prevention
- Rate limiting
- Secure token management

## 💡 Usage Tips

1. **Model Selection**: Choose between DeepSeek-V3 (powerful) or GLM-4.7-Flash (fast)
2. **Quick Actions**: Use sidebar buttons for common coding tasks
3. **Code Highlighting**: Responses automatically format code blocks
4. **New Chat**: Clear conversation history anytime

## 🐛 Troubleshooting

### Common Issues

**"HF_TOKEN not found"**
- Make sure `.env` file exists and contains your actual Hugging Face token

**"API connection failed"**
- Verify your Hugging Face token is valid
- Check your internet connection
- Try using the backup token

**"Dependencies not found"**
- Run `python setup.py` to install all requirements

## 📊 Cost Management

The app tracks token usage and estimated costs:
- DeepSeek-V3: ~$0.00002 per token
- GLM-4.7-Flash: ~$0.000001 per token

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the setup script output
3. Ensure all environment variables are configured

---

**Made with ❤️ by AMEK**