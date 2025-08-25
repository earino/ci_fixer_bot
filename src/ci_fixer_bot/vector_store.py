"""
Vector storage and similarity search for issue embeddings.

This module provides efficient storage and retrieval of embeddings using
vector similarity search. It includes an abstract base class and concrete
implementations using FAISS and in-memory storage.
"""

import json
import logging
import pickle
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .embedding_providers import EmbeddingProvider, normalize_embedding

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract base class for vector storage and similarity search.
    
    Provides efficient storage and retrieval of embeddings with associated
    metadata, supporting similarity search operations.
    """
    
    @abstractmethod
    def __init__(self, dimension: int):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of vectors to store
        """
        self.dimension = dimension
        self._lock = threading.RLock()  # Thread safety
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings with associated metadata.
        
        Args:
            embeddings: 2D array of embeddings (n_samples, dimension)
            metadata: List of metadata dicts for each embedding
            
        Raises:
            ValueError: If dimensions don't match or inputs are invalid
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold (0-1)
            
        Returns:
            List of (similarity_score, metadata) tuples, sorted by score
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the vector store to disk.
        
        Args:
            path: Path to save the store
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load from
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored vectors and metadata."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of stored vectors.
        
        Returns:
            Number of vectors in the store
        """
        pass
    
    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Validate and normalize an embedding.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            Normalized embedding
            
        Raises:
            ValueError: If embedding is invalid
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[-1]} doesn't match "
                f"store dimension {self.dimension}"
            )
        
        # Normalize for cosine similarity
        return normalize_embedding(embedding)


class FAISSVectorStore(VectorStore):
    """
    Vector store implementation using FAISS for efficient similarity search.
    
    FAISS (Facebook AI Similarity Search) provides highly optimized
    algorithms for similarity search in high-dimensional spaces.
    """
    
    def __init__(
        self, 
        dimension: int,
        use_gpu: bool = False,
        index_type: str = "flat"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of vectors
            use_gpu: Whether to use GPU acceleration if available
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        super().__init__(dimension)
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            ) from e
        
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.metadata: List[Dict[str, Any]] = []
        
        # Create appropriate index based on type
        if index_type == "flat":
            # Exact search with inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 10  # Number of clusters to search
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if requested and available
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info(f"Using GPU for FAISS operations ({faiss.get_num_gpus()} GPUs available)")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            self.on_gpu = True
        else:
            self.on_gpu = False
            if use_gpu:
                logger.warning("GPU requested but not available, using CPU")
        
        # Training flag for IVF index
        self.needs_training = index_type == "ivf"
        self.is_trained = not self.needs_training
        
        logger.info(f"Initialized FAISSVectorStore: dim={dimension}, type={index_type}, gpu={self.on_gpu}")
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings with metadata to the store."""
        with self._lock:
            # Validate inputs
            embeddings = self._validate_embedding(embeddings)
            
            if len(embeddings) != len(metadata):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) doesn't match "
                    f"metadata ({len(metadata)})"
                )
            
            # Train IVF index if needed
            if self.needs_training and not self.is_trained:
                if self.index.ntotal == 0 and len(embeddings) >= 100:
                    logger.info("Training IVF index on initial batch")
                    self.index.train(embeddings.astype(np.float32))
                    self.is_trained = True
                elif not self.is_trained:
                    # For small initial batches, fall back to adding without training
                    logger.warning(
                        f"IVF index needs at least 100 vectors for training, "
                        f"got {len(embeddings)}. Adding without training."
                    )
            
            # Add to index
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata
            self.metadata.extend(metadata)
            
            logger.debug(f"Added {len(embeddings)} vectors to store (total: {self.index.ntotal})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar vectors."""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Validate and normalize query
            query_embedding = self._validate_embedding(query_embedding)
            
            # Limit k to available vectors
            k = min(k, self.index.ntotal)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # FAISS returns -1 for not found
                    # Convert inner product to cosine similarity (both are normalized)
                    similarity = float(score)
                    
                    # Apply threshold if specified
                    if threshold is None or similarity >= threshold:
                        results.append((similarity, self.metadata[idx]))
            
            return results
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = str(path.with_suffix(".index"))
            if self.on_gpu:
                # Transfer to CPU for saving
                cpu_index = self.faiss.index_gpu_to_cpu(self.index)
                self.faiss.write_index(cpu_index, index_path)
            else:
                self.faiss.write_index(self.index, index_path)
            
            # Save metadata and config
            meta_path = str(path.with_suffix(".meta"))
            with open(meta_path, "w") as f:
                json.dump({
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "is_trained": self.is_trained,
                    "size": self.index.ntotal
                }, f)
            
            logger.info(f"Saved vector store to {path} ({self.index.ntotal} vectors)")
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        with self._lock:
            path = Path(path)
            
            # Load FAISS index
            index_path = str(path.with_suffix(".index"))
            self.index = self.faiss.read_index(index_path)
            
            # Move to GPU if needed
            if self.use_gpu and self.faiss.get_num_gpus() > 0:
                self.index = self.faiss.index_cpu_to_gpu(
                    self.faiss.StandardGpuResources(), 0, self.index
                )
                self.on_gpu = True
            
            # Load metadata and config
            meta_path = str(path.with_suffix(".meta"))
            with open(meta_path, "r") as f:
                data = json.load(f)
                self.metadata = data["metadata"]
                self.dimension = data["dimension"]
                self.index_type = data.get("index_type", "flat")
                self.is_trained = data.get("is_trained", True)
            
            logger.info(f"Loaded vector store from {path} ({self.index.ntotal} vectors)")
    
    def clear(self) -> None:
        """Clear all stored vectors."""
        with self._lock:
            self.index.reset()
            self.metadata.clear()
            self.is_trained = not self.needs_training
            logger.debug("Cleared vector store")
    
    def size(self) -> int:
        """Get the number of stored vectors."""
        return self.index.ntotal


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing and small datasets.
    
    This implementation uses numpy for all operations and doesn't require
    any external dependencies. Suitable for testing and datasets < 10,000 vectors.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize in-memory vector store.
        
        Args:
            dimension: Dimension of vectors
        """
        super().__init__(dimension)
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        logger.info(f"Initialized InMemoryVectorStore: dim={dimension}")
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings with metadata to the store."""
        with self._lock:
            # Validate inputs
            embeddings = self._validate_embedding(embeddings)
            
            if len(embeddings) != len(metadata):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) doesn't match "
                    f"metadata ({len(metadata)})"
                )
            
            # Add to store
            if self.embeddings is None:
                self.embeddings = embeddings.astype(np.float32)
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
            
            self.metadata.extend(metadata)
            
            logger.debug(f"Added {len(embeddings)} vectors to store (total: {len(self.metadata)})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar vectors using cosine similarity."""
        with self._lock:
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            
            # Validate and normalize query
            query_embedding = self._validate_embedding(query_embedding).flatten()
            
            # Compute cosine similarities (dot product of normalized vectors)
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k indices
            k = min(k, len(similarities))
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            # Prepare results
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                
                # Apply threshold if specified
                if threshold is None or similarity >= threshold:
                    results.append((similarity, self.metadata[idx]))
            
            return results
    
    def save(self, path: str) -> None:
        """Save the vector store to disk using pickle."""
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "dimension": self.dimension
            }
            
            with open(path, "wb") as f:
                pickle.dump(data, f)
            
            size = len(self.metadata) if self.metadata else 0
            logger.info(f"Saved vector store to {path} ({size} vectors)")
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        with self._lock:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            
            size = len(self.metadata) if self.metadata else 0
            logger.info(f"Loaded vector store from {path} ({size} vectors)")
    
    def clear(self) -> None:
        """Clear all stored vectors."""
        with self._lock:
            self.embeddings = None
            self.metadata.clear()
            logger.debug("Cleared vector store")
    
    def size(self) -> int:
        """Get the number of stored vectors."""
        return len(self.metadata)


def create_vector_store(
    store_type: str,
    dimension: int,
    **kwargs
) -> VectorStore:
    """
    Factory function to create vector stores.
    
    Args:
        store_type: Type of store ('faiss', 'memory')
        dimension: Dimension of vectors
        **kwargs: Additional arguments for the store
        
    Returns:
        VectorStore instance
        
    Raises:
        ValueError: If store type is unknown
    """
    if store_type == "faiss":
        return FAISSVectorStore(dimension, **kwargs)
    elif store_type == "memory":
        return InMemoryVectorStore(dimension)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")