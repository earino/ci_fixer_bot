"""
Local embedding provider using sentence-transformers.

This provider runs embedding models locally without requiring any external API.
It uses the sentence-transformers library which provides state-of-the-art
embedding models that run efficiently on CPU.
"""

import logging
from typing import List, Optional

import numpy as np

from .embedding_providers import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    This provider downloads and runs models locally, providing fast
    embeddings without external dependencies after initial model download.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize the local embedding provider.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is all-mpnet-base-v2 which provides excellent
                       quality with 768-dimensional embeddings.
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect)
            normalize_embeddings: Whether to normalize embeddings to unit length
            show_progress_bar: Whether to show progress bar for batch operations
            batch_size: Batch size for encoding multiple texts
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        
        # Lazy load the model
        self._model = None
        self._embedding_dim = None
        
        logger.info(f"Initialized LocalEmbeddingProvider with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers is required for LocalEmbeddingProvider. "
                "Install with: pip install sentence-transformers"
            ) from e
        
        logger.info(f"Loading model {self.model_name}...")
        
        try:
            # Load model with optimizations
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Set to eval mode for inference
            self._model.eval()
            
            # Get embedding dimension by encoding a dummy text
            dummy_embedding = self._model.encode(
                "dummy", 
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            self._embedding_dim = len(dummy_embedding)
            
            # CPU optimizations
            if self.device in [None, 'cpu']:
                # Enable CPU optimizations if torch is available
                try:
                    import torch
                    if hasattr(torch, 'set_num_threads'):
                        # Use all available CPU cores
                        import os
                        num_cores = os.cpu_count() or 4
                        torch.set_num_threads(num_cores)
                        logger.info(f"Using {num_cores} CPU threads for inference")
                except ImportError:
                    # Torch not available, skip CPU optimization
                    pass
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {str(e)}") from e
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array of the embedding
        """
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Encode single text
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,  # No progress for single text
                batch_size=1
            )
            
            # Ensure it's 1D
            if len(embedding.shape) > 1:
                embedding = embedding.squeeze()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {str(e)}") from e
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D numpy array where each row is an embedding
        """
        if not texts:
            return np.array([])
        
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Log batch processing for large batches
            if len(texts) > 100:
                logger.info(f"Embedding batch of {len(texts)} texts...")
            
            # Encode batch with progress bar for large batches
            show_progress = self.show_progress_bar and len(texts) > 10
            
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=show_progress,
                batch_size=self.batch_size
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to embed batch: {str(e)}") from e
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            The dimension of embeddings produced by this model
        """
        # Load model if needed to get dimension
        self._load_model()
        return self._embedding_dim
    
    def unload_model(self):
        """
        Unload the model from memory.
        
        Useful for freeing memory when the provider won't be used for a while.
        """
        if self._model is not None:
            logger.info(f"Unloading model {self.model_name}")
            del self._model
            self._model = None
            
            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"LocalEmbeddingProvider(model='{self.model_name}', "
            f"device={self.device}, loaded={self.is_loaded})"
        )


# Common model recommendations
RECOMMENDED_MODELS = {
    "all-mpnet-base-v2": {
        "description": "Best overall quality/speed balance",
        "dimension": 768,
        "size_mb": 420,
        "speed": "fast",
        "quality": "excellent"
    },
    "all-MiniLM-L6-v2": {
        "description": "5x faster, still good quality",
        "dimension": 384,
        "size_mb": 80,
        "speed": "very fast",
        "quality": "good"
    },
    "all-distilroberta-v1": {
        "description": "Good for longer texts",
        "dimension": 768,
        "size_mb": 290,
        "speed": "fast",
        "quality": "very good"
    },
    "multi-qa-mpnet-base-dot-v1": {
        "description": "Optimized for semantic search",
        "dimension": 768,
        "size_mb": 420,
        "speed": "fast",
        "quality": "excellent for search"
    },
}


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    # Remove org prefix if present
    short_name = model_name.split("/")[-1]
    
    return RECOMMENDED_MODELS.get(
        short_name,
        {
            "description": "Custom model",
            "dimension": "unknown",
            "size_mb": "unknown",
            "speed": "unknown",
            "quality": "unknown"
        }
    )