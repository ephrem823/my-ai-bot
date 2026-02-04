import os
import requests
import urllib.parse
from typing import Dict, Optional

class GoogleOAuth:
    def __init__(self):
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Missing Google OAuth credentials in environment variables")
    
    def get_auth_url(self) -> str:
        """Generate Google OAuth authorization URL"""
        # Check if credentials are properly configured
        if self.client_id == 'your_google_client_id' or not self.client_id:
            raise ValueError("Google OAuth not configured. Please set GOOGLE_CLIENT_ID in .env file")
            
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        return f"https://accounts.google.com/o/oauth2/auth?{urllib.parse.urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post('https://oauth2.googleapis.com/token', data=data)
        response.raise_for_status()
        return response.json()
    
    def get_user_info(self, access_token: str) -> Dict:
        """Get user information using access token"""
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers=headers)
        response.raise_for_status()
        return response.json()
    
    def verify_token(self, access_token: str) -> Optional[Dict]:
        """Verify access token and return user info"""
        try:
            return self.get_user_info(access_token)
        except:
            return None