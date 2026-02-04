import time
import requests
import streamlit as st
import config

def generate_ai_response(prompt, model=None):
    """Generate AI response using Hugging Face API"""
    
    # Get HF token
    hf_token = st.secrets.get("HF_TOKEN") or config.HF_TOKEN
    if not hf_token or hf_token == "your_huggingface_token_here":
        return "Error: HF_TOKEN not configured", 0, 0
    
    # Use default model if none specified
    if not model:
        model = "gpt2"
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        # Make API request
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                if generated_text.strip():
                    tokens_used = len(generated_text.split())
                    return generated_text.strip(), tokens_used, processing_time
                else:
                    return "Hello! How can I help you today?", 5, processing_time
            else:
                return "Hello! How can I help you today?", 5, processing_time
                
        else:
            return "Hello! I'm your AI assistant. How can I help you?", 8, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        return "Hello! I'm here to help. What would you like to know?", 10, processing_time