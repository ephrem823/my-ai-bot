import time
import requests
import streamlit as st
import config
from huggingface_hub import InferenceClient

def generate_ai_response(prompt, model=None):
    """Generate AI response using Hugging Face Inference Client"""
    
    # Get HF token
    hf_token = st.secrets.get("HF_TOKEN") or config.HF_TOKEN
    if not hf_token or hf_token == "your_huggingface_token_here":
        return "Error: HF_TOKEN not configured", 0, 0
    
    # Use default model if none specified
    if not model:
        model = "microsoft/DialoGPT-medium"  # Use a reliable model
    
    # Start timing
    start_time = time.time()
    
    try:
        # Use Hugging Face Inference Client
        client = InferenceClient(token=hf_token)
        
        # Generate response
        response = client.text_generation(
            prompt,
            model=model,
            max_new_tokens=1000,
            temperature=0.7,
            return_full_text=False
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if response:
            tokens_used = len(response.split())  # Approximate token count
            return response, tokens_used, processing_time
        else:
            return "Error: No response generated", 0, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        return f"Error: {str(e)}", 0, processing_time