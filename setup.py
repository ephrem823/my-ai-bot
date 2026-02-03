#!/usr/bin/env python3
"""
AMEK AI Bot Setup Script
This script helps you set up your AI bot environment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("📝 Creating .env from template...")
        
        # Copy from .env.example
        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created from template")
        else:
            print("❌ .env.example not found")
            return False
    
    # Check for required variables
    required_vars = ["HF_TOKEN"]
    missing_vars = []
    
    with open(env_path, 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=your_" in content or f"{var}=" not in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Please update these variables in .env file:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✅ .env file configured")
    return True

def test_huggingface_connection():
    """Test Hugging Face API connection"""
    try:
        from dotenv import load_dotenv
        from huggingface_hub import InferenceClient
        
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        if not token or token.startswith("your_"):
            print("⚠️  HF_TOKEN not configured in .env file")
            return False
        
        client = InferenceClient(api_key=token)
        # Simple test call
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="deepseek-ai/DeepSeek-V3",
            max_tokens=10
        )
        
        print("✅ Hugging Face API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face API connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AMEK AI Bot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check environment file
    env_ok = check_env_file()
    
    # Test API connection if env is configured
    if env_ok:
        api_ok = test_huggingface_connection()
    else:
        api_ok = False
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"✅ Python: Compatible")
    print(f"✅ Dependencies: Installed")
    print(f"{'✅' if env_ok else '⚠️ '} Environment: {'Configured' if env_ok else 'Needs configuration'}")
    print(f"{'✅' if api_ok else '⚠️ '} API: {'Connected' if api_ok else 'Not tested/failed'}")
    
    if env_ok and api_ok:
        print("\n🎉 Setup complete! You can now run:")
        print("   streamlit run app.py")
        print("   or")
        print("   streamlit run app_minimal.py")
    else:
        print("\n⚠️  Please complete the configuration steps above")

if __name__ == "__main__":
    main()