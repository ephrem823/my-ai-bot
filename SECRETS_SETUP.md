# Streamlit Secrets Setup

## For Streamlit Cloud:
1. Go to your app dashboard
2. Click "Settings" → "Secrets"
3. Add these secrets:

```toml
HF_TOKEN = "your_actual_hugging_face_token"
GOOGLE_CLIENT_ID = "your_actual_google_client_id"
GOOGLE_CLIENT_SECRET = "your_actual_google_client_secret"
GOOGLE_REDIRECT_URI = "https://your-app-name.streamlit.app"
```

## For Local Development:
Update `.streamlit/secrets.toml` with your actual credentials.

**Note:** Never commit secrets.toml to git - it's already in .gitignore