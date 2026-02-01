"""Secure Hugging Face API Integration"""

import os
import requests
from typing import Optional, Dict, Any
import streamlit as st
from security import security_manager

class SecureHuggingFace:
    def __init__(self):
        self.primary_token = os.getenv("HF_TOKEN")
        self.secondary_token = os.getenv("HF_TOKEN_SECONDARY")
        self.max_tokens = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2500"))
        self.api_url = "https://api-inference.huggingface.co/models"
        
    def _get_headers(self, use_secondary: bool = False) -> Dict[str, str]:
        """Get secure headers with token"""
        token = self.secondary_token if use_secondary else self.primary_token
        if not token or token.startswith("your_"):
            raise ValueError("Hugging Face token not configured")
        return {"Authorization": f"Bearer {token}"}
    
    def _validate_input(self, text: str) -> str:
        """Validate and sanitize input"""
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty input not allowed")
        
        # Sanitize input
        sanitized = security_manager.sanitize_input(text)
        
        # Check token limit
        if len(sanitized.split()) > self.max_tokens:
            raise ValueError(f"Input exceeds {self.max_tokens} tokens")
        
        return sanitized
    
    def generate_text(self, prompt: str, model: str = "microsoft/DialoGPT-medium") -> Optional[str]:
        """Securely generate text using Hugging Face"""
        try:
            # Validate input
            clean_prompt = self._validate_input(prompt)
            
            # Prepare request
            url = f"{self.api_url}/{model}"
            headers = self._get_headers()
            payload = {"inputs": clean_prompt}
            
            # Make request with timeout
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                # Model loading, try secondary token
                headers = self._get_headers(use_secondary=True)
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(clean_prompt, "").strip()
            
            return "I'm processing your request. Please try again."
            
        except Exception as e:
            st.error(f"AI service temporarily unavailable: {str(e)}")
            return None

# Global instance
hf_client = SecureHuggingFace()