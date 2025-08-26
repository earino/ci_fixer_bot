"""
OpenAI embedding provider for ci_fixer_bot.

Uses OpenAI's text-embedding models for high-quality embeddings.
Requires an OpenAI API key.

Installation:
1. Get an API key from https://platform.openai.com/
2. Set environment variable: export OPENAI_API_KEY=sk-...
3. Configure ci_fixer_bot to use this provider

Configuration example:
    embedding:
        provider: openai
        openai_api_key: ${OPENAI_API_KEY}  # or hardcode (not recommended)
        openai_model: text-embedding-3-small  # or text-embedding-3-large
"""

import logging
import os
from typing import List, Optional

import numpy as np
import requests

from .embedding_providers import EmbeddingProvider, EmbeddingError, normalize_embedding

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using their API.
    
    OpenAI offers state-of-the-art embedding models with excellent
    semantic understanding. Best for production deployments where
    quality is paramount.
    
    Available models:
    - text-embedding-3-small: 1536-dim, good balance of cost/quality (default)
    - text-embedding-3-large: 3072-dim, highest quality
    - text-embedding-ada-002: 1536-dim, legacy model
    
    Features:
    - State-of-the-art quality
    - Automatic retries with exponential backoff
    - Batch API for efficiency
    - Token counting for cost estimation
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Model name (default: text-embedding-3-small)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.openai.com/v1"
        
        # Model dimensions
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        # Test API access
        if not self._test_api():
            raise EmbeddingError("Failed to connect to OpenAI API")
        
        logger.info(f"Initialized OpenAIEmbeddingProvider with model: {self.model}")
    
    def _test_api(self) -> bool:
        """Test if OpenAI API is accessible with the provided key."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Test with a minimal embedding request
            payload = {
                "model": self.model,
                "input": "test"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 401:
                logger.error("Invalid OpenAI API key")
                return False
            elif response.status_code == 404:
                logger.error(f"Model {self.model} not found")
                return False
            elif response.status_code == 200:
                return True
            else:
                logger.warning(f"Unexpected response: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.warning(f"OpenAI API test failed: {e}")
            return False
    
    def _make_request(self, texts: List[str]) -> dict:
        """
        Make embedding request with retry logic.
        
        Args:
            texts: Texts to embed
            
        Returns:
            API response dictionary
            
        Raises:
            EmbeddingError: If all retries fail
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts if isinstance(texts, list) else [texts]
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
        
        raise EmbeddingError(f"OpenAI API request failed after {self.max_retries} attempts: {last_error}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using OpenAI.
        
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
            data = self._make_request([text])
            
            # Extract embedding from response
            if "data" in data and len(data["data"]) > 0:
                embedding = np.array(
                    data["data"][0]["embedding"],
                    dtype=np.float32
                )
                
                # Log token usage for cost tracking
                if "usage" in data:
                    logger.debug(f"Tokens used: {data['usage'].get('total_tokens', 0)}")
                
                return self.normalize_embedding(embedding)
            else:
                raise EmbeddingError(f"Invalid response format from OpenAI")
                
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        OpenAI's API supports batch embedding natively.
        
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
        
        # OpenAI has a limit on batch size, split if necessary
        max_batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(valid_texts), max_batch_size):
            batch = valid_texts[i:i + max_batch_size]
            
            try:
                data = self._make_request(batch)
                
                # Extract embeddings, maintaining order
                if "data" in data:
                    # Sort by index to ensure correct order
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    batch_embeddings = np.array(
                        [item["embedding"] for item in sorted_data],
                        dtype=np.float32
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Log token usage
                    if "usage" in data:
                        logger.debug(
                            f"Batch {i//max_batch_size + 1}: "
                            f"{data['usage'].get('total_tokens', 0)} tokens"
                        )
                else:
                    raise EmbeddingError("Invalid batch response format")
                    
            except Exception as e:
                raise EmbeddingError(f"Batch embedding failed: {e}")
        
        # Concatenate all batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            return self.normalize_embedding(embeddings)
        else:
            return np.array([], dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension based on model
        """
        return self._model_dimensions.get(self.model, 1536)
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self._test_api()
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "provider": "openai",
            "dimension": self.get_embedding_dim(),
            "api_base": self.base_url,
            "features": {
                "batch_support": True,
                "max_batch_size": 100,
                "retry_logic": True,
                "token_counting": True
            }
        }
    
    def estimate_cost(self, num_tokens: int) -> float:
        """
        Estimate the cost for embedding a given number of tokens.
        
        Args:
            num_tokens: Number of tokens to embed
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
            "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
            "text-embedding-ada-002": 0.10   # $0.10 per 1M tokens
        }
        
        price_per_million = pricing.get(self.model, 0.02)
        return (num_tokens / 1_000_000) * price_per_million