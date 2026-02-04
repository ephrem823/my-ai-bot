import asyncio
import aiohttp
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from datetime import datetime, timedelta
import config

class OptimizedAIManager:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.session = None
        self.rate_limiter = {}
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_data = f"{prompt}_{model}_{kwargs.get('temperature', 0.7)}_{kwargs.get('max_tokens', 1000)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - timestamp < timedelta(seconds=self.cache_ttl)
    
    def _check_rate_limit(self, model: str) -> bool:
        """Simple rate limiting check"""
        now = time.time()
        if model not in self.rate_limiter:
            self.rate_limiter[model] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limiter[model] = [
            req_time for req_time in self.rate_limiter[model] 
            if now - req_time < 60
        ]
        
        # Check if under limit (max 10 requests per minute)
        if len(self.rate_limiter[model]) >= 10:
            return False
        
        self.rate_limiter[model].append(now)
        return True
    
    async def generate_response_async(self, prompt: str, model: str, **kwargs) -> Tuple[str, int, float]:
        """Generate AI response with async processing and caching"""
        start_time = time.time()
        cache_key = self._get_cache_key(prompt, model, **kwargs)
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                processing_time = time.time() - start_time
                return cached_data[0], cached_data[1], processing_time
        
        # Check rate limiting
        if not self._check_rate_limit(model):
            return "Rate limit exceeded. Please wait a moment.", 0, time.time() - start_time
        
        try:
            # Make async API call
            response = await self._make_async_api_call(prompt, model, **kwargs)
            processing_time = time.time() - start_time
            tokens_used = len(prompt.split()) + len(response.split())
            
            result = (response, tokens_used, processing_time)
            
            # Cache the result
            self.cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error: {str(e)}"
            return error_msg, 0, processing_time
    
    async def _make_async_api_call(self, prompt: str, model: str, **kwargs) -> str:
        """Make async API call to Hugging Face"""
        session = await self.get_session()
        
        headers = {
            "Authorization": f"Bearer {config.HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False,
                "do_sample": True,
                "top_p": kwargs.get("top_p", 0.9)
            }
        }
        
        model_url = f"https://api-inference.huggingface.co/models/{model}"
        
        async with session.post(model_url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated")
                return str(result)
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    def generate_multiple_responses(self, prompts: List[str], model: str, **kwargs) -> List[Tuple[str, int, float]]:
        """Generate multiple responses concurrently"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tasks = [
                self.generate_response_async(prompt, model, **kwargs) 
                for prompt in prompts
            ]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            return results
        finally:
            loop.close()
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(
            1 for _, (_, timestamp) in self.cache.items() 
            if self._is_cache_valid(timestamp)
        )
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / max(len(self.cache), 1)
        }
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.executor.shutdown(wait=True)

# Backup AI Manager for fallback
class BackupAIManager:
    def __init__(self):
        self.backup_models = [
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill"
        ]
    
    def generate_simple_response(self, prompt: str) -> Tuple[str, int, float]:
        """Generate simple fallback response"""
        start_time = time.time()
        
        # Simple rule-based responses for common queries
        responses = {
            "hello": "Hello! I'm here to help you with coding questions.",
            "help": "I can help you with Python, JavaScript, debugging, and code optimization.",
            "error": "I'd be happy to help debug your code. Please share the error message and code.",
            "function": "I can help you create functions. What functionality do you need?",
        }
        
        prompt_lower = prompt.lower()
        for key, response in responses.items():
            if key in prompt_lower:
                processing_time = time.time() - start_time
                return response, len(response.split()), processing_time
        
        # Default response
        default_response = "I'm experiencing some technical difficulties. Could you please rephrase your question?"
        processing_time = time.time() - start_time
        return default_response, len(default_response.split()), processing_time