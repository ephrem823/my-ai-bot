import os
import requests
from urllib.parse import urlencode
import streamlit as st

class GoogleOAuth:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET") 
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")
        self.is_configured = bool(self.client_id and self.client_secret)
    
    def get_auth_url(self):
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline"
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
        response = requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers=headers)
        return response.json()