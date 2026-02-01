import streamlit as st
import requests
import json
from urllib.parse import urlencode
import os
from dotenv import load_dotenv

load_dotenv()

class GoogleAuth:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")
        
    def get_auth_url(self):
        """Generate Google OAuth URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
    
    def get_user_info(self, code):
        """Exchange code for user info"""
        # Get access token
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        token_response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
        token_json = token_response.json()
        
        if 'access_token' not in token_json:
            return None
            
        # Get user info
        headers = {'Authorization': f"Bearer {token_json['access_token']}"}
        user_response = requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers=headers)
        
        return user_response.json()

def init_google_auth():
    """Initialize Google authentication in session state"""
    if 'google_auth' not in st.session_state:
        st.session_state.google_auth = GoogleAuth()
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Check for OAuth callback
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.user:
        code = query_params['code']
        user_info = st.session_state.google_auth.get_user_info(code)
        
        if user_info:
            st.session_state.user = {
                'email': user_info['email'],
                'name': user_info['name'],
                'picture': user_info.get('picture', ''),
                'is_logged_in': True
            }
            st.rerun()

def login_button():
    """Display Google login button"""
    auth_url = st.session_state.google_auth.get_auth_url()
    st.markdown(f'<a href="{auth_url}" target="_self"><button style="background:#4285f4;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer;">🔐 Login with Google</button></a>', unsafe_allow_html=True)

def logout():
    """Logout user"""
    st.session_state.user = None
    st.rerun()

def is_logged_in():
    """Check if user is logged in"""
    return st.session_state.user and st.session_state.user.get('is_logged_in', False)

def get_user():
    """Get current user"""
    return st.session_state.user if is_logged_in() else None