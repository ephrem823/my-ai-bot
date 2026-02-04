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
        model = "gpt2"  # Use GPT-2 as it's more reliable
    
    # Start timing
    start_time = time.time()
    
    try:
        # Use Hugging Face Inference Client
        client = InferenceClient(token=hf_token)
        
        # Generate response with better error handling
        response = client.text_generation(
            prompt,
            model=model,
            max_new_tokens=500,
            temperature=0.7,
            return_full_text=False,
            do_sample=True
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if response and response.strip():
            tokens_used = len(response.split())  # Approximate token count
            return response.strip(), tokens_used, processing_time
        else:
            return "I'm having trouble generating a response right now. Please try again.", 0, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "Rate limit exceeded. Please wait a moment and try again.", 0, processing_time
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return "Model temporarily unavailable. Please try again later.", 0, processing_time
        else:
            return f"I encountered an error: {error_msg}", 0, processing_time