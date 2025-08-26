"""
LM Studio embedding provider for ci_fixer_bot.

Connects to LM Studio's OpenAI-compatible API for local embeddings.
LM Studio allows running any GGUF embedding model locally with GPU acceleration.

Installation:
1. Download LM Studio: https://lmstudio.ai/
2. Load an embedding model (e.g., nomic-embed-text, bge-small, etc.)
3. Start the local server (default: http://localhost:1234)
4. Configure ci_fixer_bot to use this provider

Configuration example:
    embedding:
        provider: lm-studio
        lm_studio_url: http://localhost:1234/v1
        model: nomic-ai/nomic-embed-text-v1.5-GGUF  # or whatever model you loaded
"""

import logging
from typing import List, Optional, Union

import numpy as np
import requests

from .embedding_providers import EmbeddingProvider, EmbeddingError, normalize_embedding

logger = logging.getLogger(__name__)


class LMStudioEmbeddingProvider(EmbeddingProvider):
    """
    LM Studio embedding provider using OpenAI-compatible API.
    
    LM Studio provides a local server that mimics OpenAI's API, allowing
    you to run any GGUF embedding model on your hardware with GPU acceleration.
    
    Features:
    - Completely local and private
    - GPU acceleration support
    - Wide model compatibility
    - No API keys required
    - OpenAI-compatible interface
    """
    
    def __init__(
        self,
        url: str = "http://localhost:1234/v1",
        model: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize LM Studio embedding provider.
        
        Args:
            url: LM Studio server URL (default: http://localhost:1234/v1)
            model: Model name (optional, uses LM Studio's loaded model)
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._embedding_dim = None
        
        # Test connection
        if not self._test_connection():
            raise EmbeddingError(
                f"Cannot connect to LM Studio at {self.url}. "
                "Please ensure LM Studio is running with an embedding model loaded."
            )
        
        logger.info(f"Initialized LMStudioEmbeddingProvider with URL: {self.url}")
    
    def _test_connection(self) -> bool:
        """Test if LM Studio server is accessible."""
        try:
            response = requests.get(
                f"{self.url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            models = response.json()
            if models.get("data"):
                available_models = [m["id"] for m in models["data"]]
                logger.info(f"Available models in LM Studio: {available_models}")
                
                # If no model specified, use the first available
                if not self.model and available_models:
                    self.model = available_models[0]
                    logger.info(f"Using model: {self.model}")
                
                return True
            return False
            
        except requests.RequestException as e:
            logger.warning(f"LM Studio connection test failed: {e}")
            return False
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using LM Studio.
        
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
                "input": text,
                "model": self.model or "default"
            }
            
            response = requests.post(
                f"{self.url}/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embedding from OpenAI-format response
            if "data" in data and len(data["data"]) > 0:
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
                
                # Cache embedding dimension
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)
                    logger.info(f"Detected embedding dimension: {self._embedding_dim}")
                
                return self.normalize_embedding(embedding)
            else:
                raise EmbeddingError(f"Invalid response format from LM Studio: {data}")
                
        except requests.RequestException as e:
            raise EmbeddingError(f"LM Studio request failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise EmbeddingError(f"Failed to parse LM Studio response: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        LM Studio's OpenAI-compatible API supports batch embedding.
        
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
        
        try:
            # LM Studio accepts array of strings for batch embedding
            payload = {
                "input": valid_texts,
                "model": self.model or "default"
            }
            
            response = requests.post(
                f"{self.url}/embeddings",
                json=payload,
                timeout=self.timeout * len(valid_texts)  # Scale timeout with batch size
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embeddings from response
            if "data" in data:
                # Sort by index to ensure correct order
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings = np.array(
                    [item["embedding"] for item in sorted_data],
                    dtype=np.float32
                )
                
                # Cache embedding dimension
                if self._embedding_dim is None and len(embeddings) > 0:
                    self._embedding_dim = embeddings.shape[1]
                    logger.info(f"Detected embedding dimension: {self._embedding_dim}")
                
                return self.normalize_embedding(embeddings)
            else:
                raise EmbeddingError(f"Invalid batch response format from LM Studio")
                
        except requests.RequestException as e:
            # Fallback to sequential processing if batch fails
            logger.warning(f"Batch embedding failed, falling back to sequential: {e}")
            embeddings = []
            for text in valid_texts:
                embeddings.append(self.embed_text(text))
            return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension (e.g., 384, 768, 1536)
        """
        if self._embedding_dim is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_text("test")
            self._embedding_dim = len(test_embedding)
        
        return self._embedding_dim
    
    def is_available(self) -> bool:
        """Check if LM Studio is available."""
        return self._test_connection()
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            response = requests.get(
                f"{self.url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            models = response.json()
            for model in models.get("data", []):
                if model["id"] == self.model:
                    return {
                        "model": self.model,
                        "provider": "lm-studio",
                        "url": self.url,
                        "info": model
                    }
            
            return {
                "model": self.model or "default",
                "provider": "lm-studio",
                "url": self.url
            }
            
        except requests.RequestException:
            return {
                "model": self.model or "unknown",
                "provider": "lm-studio",
                "url": self.url,
                "status": "unavailable"
            }