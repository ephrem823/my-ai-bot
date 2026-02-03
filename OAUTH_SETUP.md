# Google OAuth Setup Guide

## 1. Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google+ API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google+ API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add authorized redirect URIs:
     - For local: `http://localhost:8501`
     - For production: `https://your-app-domain.com`

## 2. Configure Secrets

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Fill in your Google OAuth credentials:
   - `client_id`: Your Google OAuth client ID
   - `client_secret`: Your Google OAuth client secret
   - `redirect_uri`: Must match what you set in Google Console
3. Add your Hugging Face token

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the App

```bash
streamlit run app_minimal.py
```

## How It Works

- When users visit your app, they'll see a Google login button
- Clicking it redirects to Google's OAuth consent screen
- After authorization, Google redirects back with an auth code
- The app exchanges this code for user info and creates a session
- Users stay logged in until they click logout or session expires

## Security Features

- Secure OAuth 2.0 flow
- Session-based authentication
- Input sanitization
- No passwords stored locally