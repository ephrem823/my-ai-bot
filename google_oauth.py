import os
import requests
from urllib.parse import urlencode
import streamlit as st

class GoogleOAuth:
    def __init__(self):
        # 1. Try to get secrets from the [auth] section (Recommended)
        auth_secrets = st.secrets.get("auth", {})
        
        self.client_id = auth_secrets.get("client_id") or st.secrets.get("GOOGLE_CLIENT_ID")
        self.client_secret = auth_secrets.get("client_secret") or st.secrets.get("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = auth_secrets.get("redirect_uri") or st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
        
        # 2. Set configuration status
        self.is_configured = bool(self.client_id and self.client_secret)
    
    def get_auth_url(self):
        if not self.is_configured:
            return None
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "prompt": "select_account" # Forces account selection to avoid auto-login loops
        }
        return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
    
    def exchange_code_for_token(self, code):
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        response = requests.post("https://oauth2.googleapis.com/token", data=data)
        return response.json()
    
    def get_user_info(self, access_token):
        headers = {"Authorization": f"Bearer {access_token}"}
        # Using the v3 endpoint for better data consistency
        response = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
        return response.json()
