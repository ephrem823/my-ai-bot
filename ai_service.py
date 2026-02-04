import time
import requests
import streamlit as st
import config

def get_coding_response(prompt):
    """Generate coding responses for common requests"""
    prompt_lower = prompt.lower()
    
    if "ui" in prompt_lower and "chatbot" in prompt_lower and "python" in prompt_lower:
        return '''Here's a simple Python chatbot UI using tkinter:

```python
import tkinter as tk
from tkinter import scrolledtext

class ChatbotUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chatbot")
        self.window.geometry("500x600")
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.window, 
            wrap=tk.WORD, 
            width=60, 
            height=25,
            state=tk.DISABLED
        )
        self.chat_display.pack(pady=10)
        
        # Input frame
        input_frame = tk.Frame(self.window)
        input_frame.pack(pady=5)
        
        # Message input
        self.message_entry = tk.Entry(input_frame, width=40)
        self.message_entry.pack(side=tk.LEFT, padx=5)
        self.message_entry.bind("<Return>", self.send_message)
        
        # Send button
        send_button = tk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message
        )
        send_button.pack(side=tk.LEFT)
        
    def send_message(self, event=None):
        message = self.message_entry.get()
        if message.strip():
            self.display_message(f"You: {message}")
            
            # Simple bot response
            bot_response = self.get_bot_response(message)
            self.display_message(f"Bot: {bot_response}")
            
            self.message_entry.delete(0, tk.END)
    
    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def get_bot_response(self, message):
        # Simple responses
        if "hello" in message.lower():
            return "Hello! How can I help you?"
        elif "how are you" in message.lower():
            return "I'm doing well, thank you!"
        else:
            return "That's interesting! Tell me more."
    
    def run(self):
        self.window.mainloop()

# Run the chatbot
if __name__ == "__main__":
    chatbot = ChatbotUI()
    chatbot.run()
```

This creates a simple chatbot window with:
- Scrollable chat area
- Text input field
- Send button
- Basic response logic''', 150, 0.1
    
    elif "ui" in prompt_lower and "python" in prompt_lower:
        return '''Here's a basic Python UI template using tkinter:

```python
import tkinter as tk
from tkinter import ttk

class SimpleUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("My App")
        self.root.geometry("400x300")
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Label
        label = tk.Label(self.root, text="Hello World!")
        label.pack(pady=10)
        
        # Entry
        self.entry = tk.Entry(self.root)
        self.entry.pack(pady=5)
        
        # Button
        button = tk.Button(
            self.root, 
            text="Click Me", 
            command=self.button_click
        )
        button.pack(pady=5)
    
    def button_click(self):
        text = self.entry.get()
        print(f"You entered: {text}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleUI()
    app.run()
```''', 80, 0.1
    
    return None

def generate_ai_response(prompt, model=None):
    """Generate AI response using Hugging Face API with coding fallbacks"""
    
    # Check for coding responses first
    coding_response = get_coding_response(prompt)
    if coding_response:
        return coding_response
    
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