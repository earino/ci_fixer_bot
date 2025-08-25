"""
Unit tests for vector store implementations.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from ci_fixer_bot.vector_store import (
    VectorStore,
    InMemoryVectorStore,
    FAISSVectorStore,
    create_vector_store,
)


class TestInMemoryVectorStore:
    """Test the in-memory vector store implementation."""
    
    def test_initialization(self):
        """Test store initializes correctly."""
        store = InMemoryVectorStore(dimension=768)
        
        assert store.dimension == 768
        assert store.embeddings is None
        assert store.metadata == []
        assert store.size() == 0
    
    def test_add_single_vector(self):
        """Test adding a single vector."""
        store = InMemoryVectorStore(dimension=384)
        
        embedding = np.random.randn(384)
        metadata = {"id": 1, "title": "Test issue"}
        
        store.add(embedding.reshape(1, -1), [metadata])
        
        assert store.size() == 1
        assert len(store.metadata) == 1
        assert store.metadata[0] == metadata
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors."""
        store = InMemoryVectorStore(dimension=128)
        
        embeddings = np.random.randn(5, 128)
        metadata = [{"id": i, "title": f"Issue {i}"} for i in range(5)]
        
        store.add(embeddings, metadata)
        
        assert store.size() == 5
        assert len(store.metadata) == 5
        assert store.embeddings.shape == (5, 128)
    
    def test_add_incremental(self):
        """Test adding vectors incrementally."""
        store = InMemoryVectorStore(dimension=256)
        
        # First batch
        embeddings1 = np.random.randn(3, 256)
        metadata1 = [{"id": i} for i in range(3)]
        store.add(embeddings1, metadata1)
        
        # Second batch
        embeddings2 = np.random.randn(2, 256)
        metadata2 = [{"id": i + 3} for i in range(2)]
        store.add(embeddings2, metadata2)
        
        assert store.size() == 5
        assert store.embeddings.shape == (5, 256)
    
    def test_search_basic(self):
        """Test basic similarity search."""
        store = InMemoryVectorStore(dimension=128)
        
        # Add some vectors
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Vector pointing in dimension 0
            [0.0, 1.0, 0.0, 0.0],  # Vector pointing in dimension 1
            [0.0, 0.0, 1.0, 0.0],  # Vector pointing in dimension 2
            [1.0, 1.0, 0.0, 0.0],  # Combination of 0 and 1
        ])
        # Pad to match dimension
        embeddings = np.pad(embeddings, ((0, 0), (0, 124)), mode='constant')
        
        metadata = [{"id": i} for i in range(4)]
        store.add(embeddings, metadata)
        
        # Search for vector similar to first one
        query = np.zeros(128)
        query[0] = 1.0
        
        results = store.search(query, k=2)
        
        assert len(results) == 2
        # First result should be the exact match
        assert results[0][1]["id"] == 0
        # Score should be high (normalized vectors)
        assert results[0][0] > 0.5
    
    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        store = InMemoryVectorStore(dimension=64)
        
        # Create orthogonal vectors (no similarity)
        embeddings = np.eye(5, 64)
        metadata = [{"id": i} for i in range(5)]
        store.add(embeddings, metadata)
        
        # Search for first vector with high threshold
        query = np.zeros(64)
        query[0] = 1.0
        
        results = store.search(query, k=5, threshold=0.9)
        
        # Only the exact match should pass threshold
        assert len(results) == 1
        assert results[0][1]["id"] == 0
    
    def test_search_empty_store(self):
        """Test search on empty store."""
        store = InMemoryVectorStore(dimension=768)
        
        query = np.random.randn(768)
        results = store.search(query, k=5)
        
        assert results == []
    
    def test_dimension_validation(self):
        """Test dimension validation."""
        store = InMemoryVectorStore(dimension=100)
        
        # Wrong dimension should raise error
        wrong_embedding = np.random.randn(50)
        with pytest.raises(ValueError, match="dimension.*doesn't match"):
            store.add(wrong_embedding.reshape(1, -1), [{"id": 1}])
    
    def test_metadata_count_validation(self):
        """Test metadata count validation."""
        store = InMemoryVectorStore(dimension=100)
        
        embeddings = np.random.randn(3, 100)
        metadata = [{"id": 1}]  # Only 1 metadata for 3 embeddings
        
        with pytest.raises(ValueError, match="Number of embeddings.*doesn't match"):
            store.add(embeddings, metadata)
    
    def test_normalization(self):
        """Test that vectors are normalized."""
        store = InMemoryVectorStore(dimension=128)
        
        # Add unnormalized vector
        embedding = np.ones(128) * 2  # Not normalized
        store.add(embedding.reshape(1, -1), [{"id": 1}])
        
        # Check it was normalized
        stored = store.embeddings[0]
        norm = np.linalg.norm(stored)
        assert np.isclose(norm, 1.0, rtol=1e-5)
    
    def test_save_and_load(self):
        """Test saving and loading store."""
        store = InMemoryVectorStore(dimension=256)
        
        # Add some data
        embeddings = np.random.randn(10, 256)
        metadata = [{"id": i, "data": f"test_{i}"} for i in range(10)]
        store.add(embeddings, metadata)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            store.save(temp_path)
            
            # Load into new store
            new_store = InMemoryVectorStore(dimension=256)
            new_store.load(temp_path)
            
            assert new_store.size() == 10
            assert new_store.dimension == 256
            assert len(new_store.metadata) == 10
            np.testing.assert_array_almost_equal(
                new_store.embeddings, 
                store.embeddings
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_clear(self):
        """Test clearing the store."""
        store = InMemoryVectorStore(dimension=512)
        
        # Add data
        embeddings = np.random.randn(5, 512)
        metadata = [{"id": i} for i in range(5)]
        store.add(embeddings, metadata)
        
        assert store.size() == 5
        
        # Clear
        store.clear()
        
        assert store.size() == 0
        assert store.embeddings is None
        assert store.metadata == []
    
    def test_thread_safety(self):
        """Test thread safety of operations."""
        import threading
        
        store = InMemoryVectorStore(dimension=128)
        
        def add_vectors():
            for i in range(10):
                embedding = np.random.randn(128)
                store.add(embedding.reshape(1, -1), [{"id": i}])
        
        # Run multiple threads
        threads = [threading.Thread(target=add_vectors) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have added 30 vectors total
        assert store.size() == 30


class TestFAISSVectorStore:
    """Test FAISS vector store with mocked FAISS."""
    
    def test_initialization_flat_index(self):
        """Test initialization with flat index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=768, index_type="flat")
            
            assert store.dimension == 768
            assert store.index_type == "flat"
            assert store.metadata == []
            assert store.size() == 0
            mock_faiss.IndexFlatIP.assert_called_once_with(768)
    
    def test_initialization_ivf_index(self):
        """Test initialization with IVF index."""
        mock_faiss = MagicMock()
        mock_quantizer = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.nprobe = 10
        mock_faiss.IndexFlatIP.return_value = mock_quantizer
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.METRIC_INNER_PRODUCT = 0
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=384, index_type="ivf")
            
            assert store.index_type == "ivf"
            assert store.needs_training is True
            assert store.is_trained is False
            mock_faiss.IndexIVFFlat.assert_called_once()
    
    def test_initialization_missing_faiss(self):
        """Test helpful error when FAISS not installed."""
        # Don't mock faiss, so import will fail
        store = FAISSVectorStore.__new__(FAISSVectorStore)
        store.dimension = 768
        store.metadata = []
        store._lock = threading.RLock()
        
        with pytest.raises(ImportError, match="faiss is required"):
            FAISSVectorStore.__init__(store, dimension=768)
    
    def test_add_vectors(self):
        """Test adding vectors to FAISS index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=256)
            
            embeddings = np.random.randn(3, 256)
            metadata = [{"id": i} for i in range(3)]
            
            # Mock ntotal to simulate adding
            mock_index.ntotal = 3
            
            store.add(embeddings, metadata)
            
            mock_index.add.assert_called_once()
            assert len(store.metadata) == 3
    
    def test_search_vectors(self):
        """Test searching vectors in FAISS index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        # Mock search results
        mock_scores = np.array([[0.9, 0.8, 0.7]])
        mock_indices = np.array([[0, 2, 4]])
        mock_index.search.return_value = (mock_scores, mock_indices)
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=128)
            store.metadata = [{"id": i} for i in range(5)]
            
            query = np.random.randn(128)
            results = store.search(query, k=3)
            
            assert len(results) == 3
            assert results[0][0] == 0.9  # Score
            assert results[0][1]["id"] == 0  # Metadata
            assert results[1][0] == 0.8
            assert results[1][1]["id"] == 2
    
    def test_save_and_load(self):
        """Test saving and loading FAISS index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 10
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=512)
            store.metadata = [{"id": i} for i in range(10)]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "store"
                
                # Mock file operations
                with patch("builtins.open", create=True) as mock_open:
                    with patch("json.dump") as mock_json_dump:
                        store.save(str(save_path))
                        
                        # Check FAISS write was called
                        mock_faiss.write_index.assert_called_once()
                        
                        # Check metadata was saved
                        mock_json_dump.assert_called_once()
                
                # Test loading
                with patch("builtins.open", create=True) as mock_open:
                    with patch("json.load") as mock_json_load:
                        mock_json_load.return_value = {
                            "metadata": store.metadata,
                            "dimension": 512,
                            "index_type": "flat",
                            "is_trained": True,
                            "size": 10
                        }
                        
                        new_store = FAISSVectorStore(dimension=512)
                        new_store.load(str(save_path))
                        
                        mock_faiss.read_index.assert_called_once()
    
    def test_gpu_initialization(self):
        """Test GPU acceleration initialization."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_gpu_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 2  # Has GPUs
        mock_faiss.StandardGpuResources.return_value = MagicMock()
        mock_faiss.index_cpu_to_gpu.return_value = mock_gpu_index
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=768, use_gpu=True)
            
            assert store.use_gpu is True
            assert store.on_gpu is True
            mock_faiss.index_cpu_to_gpu.assert_called_once()
    
    def test_clear(self):
        """Test clearing the FAISS store."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = FAISSVectorStore(dimension=256)
            store.metadata = [{"id": i} for i in range(5)]
            
            store.clear()
            
            mock_index.reset.assert_called_once()
            assert store.metadata == []


class TestVectorStoreFactory:
    """Test the vector store factory function."""
    
    def test_create_memory_store(self):
        """Test creating in-memory store."""
        store = create_vector_store("memory", dimension=768)
        
        assert isinstance(store, InMemoryVectorStore)
        assert store.dimension == 768
    
    def test_create_faiss_store(self):
        """Test creating FAISS store."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            store = create_vector_store(
                "faiss", 
                dimension=384,
                index_type="flat"
            )
            
            assert isinstance(store, FAISSVectorStore)
            assert store.dimension == 384
    
    def test_unknown_store_type(self):
        """Test error for unknown store type."""
        with pytest.raises(ValueError, match="Unknown vector store type"):
            create_vector_store("unknown", dimension=768)


import threading


class TestVectorStoreNormalization:
    """Test vector normalization in stores."""
    
    def test_normalize_embedding_helper(self):
        """Test the normalize_embedding helper function."""
        from ci_fixer_bot.embedding_providers import normalize_embedding
        
        # Test 1D vector
        vec1d = np.array([3.0, 4.0])
        normalized = normalize_embedding(vec1d)
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        
        # Test 2D vector
        vec2d = np.array([[3.0, 4.0], [5.0, 12.0]])
        normalized = normalize_embedding(vec2d)
        assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
        assert np.isclose(np.linalg.norm(normalized[1]), 1.0)
    
    def test_store_normalizes_on_add(self):
        """Test that stores normalize vectors on add."""
        store = InMemoryVectorStore(dimension=128)
        
        # Add unnormalized vector
        unnormalized = np.ones(128) * 5
        store.add(unnormalized.reshape(1, -1), [{"id": 1}])
        
        # Check it's normalized
        stored = store.embeddings[0]
        assert np.isclose(np.linalg.norm(stored), 1.0)
    
    def test_store_normalizes_query(self):
        """Test that stores normalize query vectors."""
        store = InMemoryVectorStore(dimension=64)
        
        # Add a normalized vector
        normalized = np.zeros(64)
        normalized[0] = 1.0
        store.add(normalized.reshape(1, -1), [{"id": 1}])
        
        # Search with unnormalized query
        query = np.ones(64) * 10
        query[0] = 100  # Make first dimension dominant
        
        results = store.search(query, k=1)
        
        # Should still find the match
        assert len(results) == 1
        assert results[0][1]["id"] == 1