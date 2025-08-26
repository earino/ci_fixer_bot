"""
Ollama embedding provider for ci_fixer_bot.

Uses Ollama for local embeddings with models like nomic-embed-text.
Ollama provides an easy way to run embedding models locally.

Installation:
1. Install Ollama: https://ollama.ai/
2. Pull an embedding model: ollama pull nomic-embed-text
3. Ollama runs automatically (default: http://localhost:11434)
4. Configure ci_fixer_bot to use this provider

Configuration example:
    embedding:
        provider: ollama
        ollama_url: http://localhost:11434
        ollama_model: nomic-embed-text  # or mxbai-embed-large
"""

import logging
from typing import List, Optional

import numpy as np
import requests

from .embedding_providers import EmbeddingProvider, EmbeddingError, normalize_embedding

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider for local embeddings.
    
    Ollama is a tool for running large language models and embedding models
    locally. It's simpler than LM Studio but equally powerful for embeddings.
    
    Recommended models:
    - nomic-embed-text: 768-dim, excellent quality (default)
    - mxbai-embed-large: 1024-dim, state-of-the-art quality
    - all-minilm: 384-dim, fast and lightweight
    
    Features:
    - Simple installation and setup
    - Automatic model management
    - GPU acceleration when available
    - RESTful API
    """
    
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: int = 30
    ):
        """
        Initialize Ollama embedding provider.
        
        Args:
            url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: nomic-embed-text)
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._embedding_dim = None
        
        # Test connection and pull model if needed
        if not self._ensure_model_available():
            raise EmbeddingError(
                f"Cannot use model {model} with Ollama at {url}. "
                "Please ensure Ollama is running and the model is available."
            )
        
        logger.info(f"Initialized OllamaEmbeddingProvider with model: {self.model}")
    
    def _ensure_model_available(self) -> bool:
        """
        Check if model is available, attempt to pull if not.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model exists
            models = response.json()
            model_names = [m["name"] for m in models.get("models", [])]
            
            if self.model in model_names or f"{self.model}:latest" in model_names:
                logger.info(f"Model {self.model} is available in Ollama")
                return True
            
            # Model not found, attempt to pull it
            logger.info(f"Model {self.model} not found, attempting to pull...")
            pull_response = requests.post(
                f"{self.url}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes for model download
            )
            
            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled model {self.model}")
                return True
            else:
                logger.warning(f"Failed to pull model {self.model}: {pull_response.text}")
                return False
                
        except requests.RequestException as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text:
            raise ValueError("Cannot embed empty text")
        
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embedding from response
            if "embedding" in data:
                embedding = np.array(data["embedding"], dtype=np.float32)
                
                # Cache embedding dimension
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)
                    logger.info(f"Detected embedding dimension: {self._embedding_dim}")
                
                return self.normalize_embedding(embedding)
            else:
                raise EmbeddingError(f"Invalid response format from Ollama: {data}")
                
        except requests.RequestException as e:
            raise EmbeddingError(f"Ollama request failed: {e}")
        except (KeyError, ValueError) as e:
            raise EmbeddingError(f"Failed to parse Ollama response: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Note: Ollama doesn't have native batch support, so we process
        sequentially but efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of normalized embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Remove empty texts
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        embeddings = []
        
        # Process texts with progress tracking for large batches
        for i, text in enumerate(valid_texts):
            if i > 0 and i % 10 == 0:
                logger.debug(f"Processed {i}/{len(valid_texts)} embeddings")
            
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except EmbeddingError as e:
                logger.warning(f"Failed to embed text {i}: {e}")
                # Use zero vector as fallback
                if self._embedding_dim:
                    embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                else:
                    raise
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension (depends on model)
        """
        if self._embedding_dim is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_text("test")
            self._embedding_dim = len(test_embedding)
        
        return self._embedding_dim
    
    def is_available(self) -> bool:
        """Check if Ollama is available with the model."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            models = response.json()
            model_names = [m["name"] for m in models.get("models", [])]
            return self.model in model_names or f"{self.model}:latest" in model_names
            
        except requests.RequestException:
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            response = requests.post(
                f"{self.url}/api/show",
                json={"name": self.model},
                timeout=5
            )
            response.raise_for_status()
            
            info = response.json()
            return {
                "model": self.model,
                "provider": "ollama",
                "url": self.url,
                "details": info.get("details", {}),
                "parameters": info.get("parameters", ""),
                "embedding_dim": self._embedding_dim
            }
            
        except requests.RequestException as e:
            return {
                "model": self.model,
                "provider": "ollama",
                "url": self.url,
                "status": "unavailable",
                "error": str(e)
            }
    
    @staticmethod
    def list_available_models(url: str = "http://localhost:11434") -> List[str]:
        """
        List all available embedding models in Ollama.
        
        Args:
            url: Ollama server URL
            
        Returns:
            List of available model names
        """
        embedding_models = [
            "nomic-embed-text",
            "mxbai-embed-large", 
            "all-minilm",
            "bge-small",
            "bge-base",
            "bge-large"
        ]
        
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json()
            installed = [m["name"] for m in models.get("models", [])]
            
            # Filter to only embedding models
            return [m for m in installed if any(em in m for em in embedding_models)]
            
        except requests.RequestException:
            return []