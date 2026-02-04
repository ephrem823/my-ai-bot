import time
import requests
import streamlit as st
import config

def generate_ai_response(prompt, model=None):
    """Generate AI response using Hugging Face API"""
    
    # Get HF token
    hf_token = st.secrets.get("HF_TOKEN") or config.HF_TOKEN
    if not hf_token:
        return "Error: HF_TOKEN not configured", 0, 0
    
    # Use default model if none specified
    if not model:
        model = list(config.MODELS.values())[0]
    
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
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # Make API request
        response = requests.post(
            f"https://router.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "No response generated")
                tokens_used = len(generated_text.split())  # Approximate token count
                return generated_text, tokens_used, processing_time
            else:
                return "Error: Unexpected response format", 0, processing_time
                
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return error_msg, 0, processing_time
            
    except requests.exceptions.Timeout:
        processing_time = time.time() - start_time
        return "Error: Request timed out", 0, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        return f"Error: {str(e)}", 0, processing_time