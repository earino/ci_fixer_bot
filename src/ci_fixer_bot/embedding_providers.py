"""
Embedding provider abstractions for ci_fixer_bot.

This module provides a pluggable architecture for different embedding providers,
similar to the LLM provider pattern. Supports local models, LM Studio, Ollama,
and cloud providers like OpenAI.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from .config import Config


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers must implement this interface to ensure
    compatibility with the deduplication system.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a vector representation.
        
        Args:
            text: The text to embed
            
        Returns:
            A numpy array representing the text embedding
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently in a batch operation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A 2D numpy array where each row is an embedding
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the dimension of embeddings produced by this provider.
        
        Returns:
            The embedding dimension (e.g., 768 for all-mpnet-base-v2)
        """
        pass
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize an embedding vector for cosine similarity.
        
        Args:
            embedding: The embedding vector to normalize
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def normalize_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of embeddings for cosine similarity.
        
        Args:
            embeddings: 2D array of embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.
    
    Generates deterministic embeddings based on text hash for testing
    the abstraction without requiring actual models.
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize mock provider.
        
        Args:
            embedding_dim: Dimension of mock embeddings
        """
        self.embedding_dim = embedding_dim
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a deterministic mock embedding from text."""
        # Use hash for deterministic but unique embeddings
        hash_val = hash(text)
        
        # Generate deterministic random vector from hash
        np.random.seed(abs(hash_val) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        return self.normalize_embedding(embedding)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for multiple texts."""
        embeddings = np.array([self.embed_text(text) for text in texts])
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return the configured embedding dimension."""
        return self.embedding_dim


def create_embedding_provider(config: Config) -> EmbeddingProvider:
    """
    Factory function to create the appropriate embedding provider.
    
    Args:
        config: Application configuration
        
    Returns:
        An initialized embedding provider
        
    Raises:
        ValueError: If the provider type is unknown or dependencies are missing
    """
    if not config.deduplication.use_embeddings:
        raise ValueError("Embeddings are not enabled in configuration")
    
    provider_type = config.deduplication.embedding.provider.lower()
    
    if provider_type == "mock":
        # Mock provider for testing
        return MockEmbeddingProvider(
            embedding_dim=config.deduplication.embedding.embedding_dim or 768
        )
    
    elif provider_type == "local":
        # Import here to avoid dependency if not using local
        try:
            from .embedding_providers_local import LocalEmbeddingProvider
            return LocalEmbeddingProvider(
                model_name=config.deduplication.embedding.model
            )
        except ImportError as e:
            raise ValueError(
                "Local embedding provider requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
    
    elif provider_type == "lm-studio":
        # Import here to avoid dependency if not using LM Studio
        try:
            from .embedding_providers_lmstudio import LMStudioEmbeddingProvider
            return LMStudioEmbeddingProvider(
                base_url=config.deduplication.embedding.lm_studio_url
            )
        except ImportError as e:
            raise ValueError(
                "LM Studio provider requires openai package. "
                "Install with: pip install openai"
            ) from e
    
    elif provider_type == "ollama":
        # Import here to avoid dependency if not using Ollama
        try:
            from .embedding_providers_ollama import OllamaEmbeddingProvider
            return OllamaEmbeddingProvider(
                model=config.deduplication.embedding.ollama_model,
                base_url=config.deduplication.embedding.ollama_url
            )
        except ImportError as e:
            raise ValueError(
                "Ollama provider requires requests package. "
                "Install with: pip install requests"
            ) from e
    
    elif provider_type == "openai":
        # Import here to avoid dependency if not using OpenAI
        try:
            from .embedding_providers_openai import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                api_key=config.deduplication.embedding.openai_api_key,
                model=config.deduplication.embedding.openai_model
            )
        except ImportError as e:
            raise ValueError(
                "OpenAI provider requires openai package. "
                "Install with: pip install openai"
            ) from e
    
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            f"Supported: local, lm-studio, ollama, openai, mock"
        )


def validate_embedding_provider(provider: EmbeddingProvider) -> bool:
    """
    Validate that an embedding provider is working correctly.
    
    Args:
        provider: The provider to validate
        
    Returns:
        True if provider is working, False otherwise
    """
    try:
        # Test single embedding
        test_text = "This is a test sentence for embedding validation."
        embedding = provider.embed_text(test_text)
        
        # Check dimension
        if len(embedding) != provider.get_embedding_dim():
            return False
        
        # Check it's normalized (for cosine similarity)
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, rtol=1e-5):
            # Try normalizing and check again
            normalized = provider.normalize_embedding(embedding)
            norm = np.linalg.norm(normalized)
            if not np.isclose(norm, 1.0, rtol=1e-5):
                return False
        
        # Test batch embedding
        test_texts = ["First test", "Second test", "Third test"]
        batch_embeddings = provider.embed_batch(test_texts)
        
        # Check shape
        if batch_embeddings.shape != (3, provider.get_embedding_dim()):
            return False
        
        return True
        
    except Exception:
        return False