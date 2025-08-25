"""
LLM provider abstraction for ci_fixer_bot.

Supports multiple LLM providers: Claude CLI, OpenAI, local models, custom endpoints.
"""

import json
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from .config import LLMConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def analyze(self, prompt: str) -> str:
        """Send prompt to LLM and return response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available/configured."""
        pass


class ClaudeCLIProvider(LLMProvider):
    """Claude CLI provider using the claude command-line tool."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def analyze(self, prompt: str) -> str:
        """Analyze using Claude CLI."""
        try:
            result = subprocess.run(
                ["claude", "chat", "--prompt", prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=120  # 2 minute timeout
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Claude CLI failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out")
        except FileNotFoundError:
            raise RuntimeError("Claude CLI not found. Install with: pip install claude-cli")
    
    def is_available(self) -> bool:
        """Check if Claude CLI is available."""
        try:
            subprocess.run(["claude", "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model or "gpt-3.5-turbo"
        self.api_key = config.api_key
        
        if not self.api_key:
            raise ValueError("OpenAI API key required")
    
    def analyze(self, prompt: str) -> str:
        """Analyze using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a senior engineer analyzing CI failures."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except requests.RequestException as e:
            raise RuntimeError(f"OpenAI API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected OpenAI API response format: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model or "codellama"
        self.endpoint = config.endpoint or "http://localhost:11434"
    
    def analyze(self, prompt: str) -> str:
        """Analyze using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=120  # Local models can be slower
            )
            response.raise_for_status()
            
            data = response.json()
            return data["response"].strip()
            
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
        except KeyError as e:
            raise RuntimeError(f"Unexpected Ollama API response format: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class CustomEndpointProvider(LLMProvider):
    """Custom endpoint provider for any OpenAI-compatible API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        if not config.endpoint:
            raise ValueError("Custom endpoint URL required")
        
        self.endpoint = config.endpoint
        self.headers = config.headers or {}
        self.model = config.model or "default"
    
    def analyze(self, prompt: str) -> str:
        """Analyze using custom endpoint."""
        # Try OpenAI-compatible format first
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a senior engineer analyzing CI failures."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Try OpenAI format
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            # Try simple response format
            elif "response" in data:
                return data["response"].strip()
            # Try text field
            elif "text" in data:
                return data["text"].strip()
            else:
                raise RuntimeError(f"Unknown response format: {data}")
                
        except requests.RequestException as e:
            raise RuntimeError(f"Custom endpoint request failed: {e}")
    
    def is_available(self) -> bool:
        """Check if custom endpoint is available."""
        try:
            # Try a simple health check
            response = requests.get(self.endpoint.replace('/completions', '/health'), 
                                  headers=self.headers, timeout=5)
            return True  # If no exception, assume it's available
        except requests.RequestException:
            return True  # Assume available since we can't really test


class CustomCLIProvider(LLMProvider):
    """Custom CLI command provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        if not config.command:
            raise ValueError("Custom command required")
        
        self.command = config.command.split()
    
    def analyze(self, prompt: str) -> str:
        """Analyze using custom CLI command."""
        try:
            result = subprocess.run(
                self.command + [prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Custom CLI failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Custom CLI timed out")
    
    def is_available(self) -> bool:
        """Check if custom CLI is available."""
        try:
            subprocess.run(self.command[:1] + ["--help"], 
                         capture_output=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False


class NoLLMProvider(LLMProvider):
    """Fallback provider that uses pattern matching instead of LLM."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the no-LLM provider (config is ignored)."""
        pass
    
    def analyze(self, prompt: str) -> str:
        """Return basic analysis without LLM."""
        return (
            "Basic pattern-matching analysis (no LLM):\n"
            "- Check logs for common error patterns\n"
            "- Categorize by error type\n"
            "- Apply basic risk assessment rules\n"
            "\nFor detailed analysis, configure an LLM provider."
        )
    
    def is_available(self) -> bool:
        """Always available as fallback."""
        return True


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    providers = {
        "claude-cli": ClaudeCLIProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "custom-endpoint": CustomEndpointProvider,
        "custom-cli": CustomCLIProvider,
        "none": NoLLMProvider,
    }
    
    provider_class = providers.get(config.provider)
    if not provider_class:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown LLM provider: {config.provider}. Available: {available}")
    
    provider = provider_class(config)
    
    # Check if provider is available
    if not provider.is_available():
        raise RuntimeError(f"LLM provider '{config.provider}' is not available or not configured properly")
    
    return provider