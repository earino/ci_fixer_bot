"""
Integration tests for vector stores with real backends.

Note: These tests require faiss-cpu to be installed and are marked as 
integration tests. They can be skipped in CI if needed.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if faiss not available
pytest.importorskip("faiss")

from ci_fixer_bot.vector_store import (
    FAISSVectorStore,
    InMemoryVectorStore,
    create_vector_store,
)


@pytest.mark.integration
class TestFAISSVectorStoreIntegration:
    """Integration tests with real FAISS."""
    
    @pytest.fixture
    def store(self):
        """Create a FAISS store for testing."""
        return FAISSVectorStore(dimension=384, index_type="flat")
    
    def test_real_faiss_operations(self, store):
        """Test full workflow with real FAISS."""
        # Add vectors
        embeddings = np.random.randn(100, 384).astype(np.float32)
        # Normalize them
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        metadata = [{"id": i, "text": f"Issue {i}"} for i in range(100)]
        store.add(embeddings, metadata)
        
        assert store.size() == 100
        
        # Search
        query = np.random.randn(384)
        query = query / np.linalg.norm(query)
        
        results = store.search(query, k=10)
        
        assert len(results) == 10
        # Results should be sorted by similarity
        scores = [r[0] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # All scores should be between -1 and 1 (cosine similarity)
        for score, _ in results:
            assert -1.0 <= score <= 1.0
    
    def test_semantic_similarity_search(self, store):
        """Test that similar vectors are found."""
        # Create some vectors with known relationships
        base_vector = np.random.randn(384)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Create similar and dissimilar vectors
        similar1 = base_vector + np.random.randn(384) * 0.1
        similar1 = similar1 / np.linalg.norm(similar1)
        
        similar2 = base_vector + np.random.randn(384) * 0.2
        similar2 = similar2 / np.linalg.norm(similar2)
        
        orthogonal = np.random.randn(384)
        orthogonal = orthogonal - np.dot(orthogonal, base_vector) * base_vector
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        
        # Add to store
        embeddings = np.vstack([base_vector, similar1, similar2, orthogonal])
        metadata = [
            {"id": 0, "type": "base"},
            {"id": 1, "type": "similar"},
            {"id": 2, "type": "similar"},
            {"id": 3, "type": "orthogonal"}
        ]
        store.add(embeddings, metadata)
        
        # Search for base vector
        results = store.search(base_vector, k=4)
        
        # First should be exact match
        assert results[0][1]["id"] == 0
        assert results[0][0] > 0.99  # Very high similarity
        
        # Next should be similar vectors
        assert results[1][1]["type"] == "similar"
        assert results[2][1]["type"] == "similar"
        
        # Orthogonal should be last with low similarity
        assert results[3][1]["type"] == "orthogonal"
        assert abs(results[3][0]) < 0.1  # Near zero similarity
    
    def test_persistence(self, store):
        """Test saving and loading FAISS index."""
        # Add data
        embeddings = np.random.randn(50, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        metadata = [{"id": i, "data": f"test_{i}"} for i in range(50)]
        store.add(embeddings, metadata)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "faiss_store"
            store.save(str(save_path))
            
            # Check files were created
            assert (save_path.with_suffix(".index")).exists()
            assert (save_path.with_suffix(".meta")).exists()
            
            # Load into new store
            new_store = FAISSVectorStore(dimension=384)
            new_store.load(str(save_path))
            
            assert new_store.size() == 50
            assert len(new_store.metadata) == 50
            
            # Search should work
            query = np.random.randn(384)
            query = query / np.linalg.norm(query)
            
            results = new_store.search(query, k=5)
            assert len(results) == 5
    
    def test_incremental_updates(self, store):
        """Test adding vectors incrementally."""
        # Add initial batch
        batch1 = np.random.randn(20, 384).astype(np.float32)
        batch1 = batch1 / np.linalg.norm(batch1, axis=1, keepdims=True)
        meta1 = [{"batch": 1, "id": i} for i in range(20)]
        store.add(batch1, meta1)
        
        assert store.size() == 20
        
        # Add second batch
        batch2 = np.random.randn(30, 384).astype(np.float32)
        batch2 = batch2 / np.linalg.norm(batch2, axis=1, keepdims=True)
        meta2 = [{"batch": 2, "id": i} for i in range(30)]
        store.add(batch2, meta2)
        
        assert store.size() == 50
        
        # Search should find from both batches
        query = batch1[0]  # Use first vector from batch1
        results = store.search(query, k=10)
        
        # Should find exact match
        assert results[0][1]["batch"] == 1
        assert results[0][1]["id"] == 0
        assert results[0][0] > 0.99
    
    def test_threshold_filtering(self, store):
        """Test similarity threshold filtering."""
        # Create vectors with varying similarities
        base = np.zeros(384)
        base[0] = 1.0
        
        # High similarity
        high_sim = base.copy()
        high_sim[1] = 0.1
        high_sim = high_sim / np.linalg.norm(high_sim)
        
        # Medium similarity
        med_sim = base.copy()
        med_sim[1] = 0.5
        med_sim = med_sim / np.linalg.norm(med_sim)
        
        # Low similarity
        low_sim = base.copy()
        low_sim[1] = 2.0
        low_sim = low_sim / np.linalg.norm(low_sim)
        
        embeddings = np.vstack([base, high_sim, med_sim, low_sim])
        metadata = [
            {"id": 0, "sim": "exact"},
            {"id": 1, "sim": "high"},
            {"id": 2, "sim": "medium"},
            {"id": 3, "sim": "low"}
        ]
        store.add(embeddings, metadata)
        
        # Search with high threshold
        results = store.search(base, k=10, threshold=0.9)
        
        # Should only get exact and high similarity
        assert len(results) <= 2
        for _, meta in results:
            assert meta["sim"] in ["exact", "high"]
    
    @pytest.mark.parametrize("index_type", ["flat", "hnsw"])
    def test_different_index_types(self, index_type):
        """Test different FAISS index types."""
        store = FAISSVectorStore(dimension=256, index_type=index_type)
        
        # Add vectors
        embeddings = np.random.randn(100, 256).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metadata = [{"id": i} for i in range(100)]
        
        store.add(embeddings, metadata)
        
        # Search
        query = embeddings[0]
        results = store.search(query, k=5)
        
        assert len(results) == 5
        # First result should be exact match
        assert results[0][1]["id"] == 0
        assert results[0][0] > 0.99
    
    def test_ivf_index_training(self):
        """Test IVF index training on large batch."""
        store = FAISSVectorStore(dimension=128, index_type="ivf")
        
        # IVF needs training with enough vectors
        assert store.needs_training is True
        assert store.is_trained is False
        
        # Add large batch to trigger training
        embeddings = np.random.randn(200, 128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metadata = [{"id": i} for i in range(200)]
        
        store.add(embeddings, metadata)
        
        # Should now be trained
        assert store.is_trained is True
        
        # Search should work
        query = embeddings[0]
        results = store.search(query, k=10)
        assert len(results) == 10
    
    def test_empty_store_operations(self, store):
        """Test operations on empty store."""
        # Search empty store
        query = np.random.randn(384)
        results = store.search(query, k=5)
        assert results == []
        
        # Size of empty store
        assert store.size() == 0
        
        # Clear empty store (should not error)
        store.clear()
        assert store.size() == 0


@pytest.mark.integration
@pytest.mark.slow
class TestVectorStorePerformance:
    """Performance tests for vector stores."""
    
    def test_large_scale_performance(self):
        """Test performance with large number of vectors."""
        store = FAISSVectorStore(dimension=768, index_type="flat")
        
        # Add 10,000 vectors
        n_vectors = 10000
        batch_size = 1000
        
        start_time = time.time()
        
        for i in range(0, n_vectors, batch_size):
            batch = np.random.randn(batch_size, 768).astype(np.float32)
            batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            metadata = [{"id": j} for j in range(i, i + batch_size)]
            store.add(batch, metadata)
        
        add_time = time.time() - start_time
        
        assert store.size() == n_vectors
        assert add_time < 10  # Should add 10k vectors in < 10 seconds
        
        # Search performance
        query = np.random.randn(768).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        start_time = time.time()
        results = store.search(query, k=100)
        search_time = time.time() - start_time
        
        assert len(results) == 100
        assert search_time < 0.1  # Should search 10k vectors in < 100ms
        
        print(f"Performance: Added {n_vectors} in {add_time:.2f}s, "
              f"searched in {search_time*1000:.2f}ms")
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        store = FAISSVectorStore(dimension=768, index_type="flat")
        
        # Add 100,000 vectors
        n_vectors = 100000
        batch_size = 10000
        
        for i in range(0, n_vectors, batch_size):
            batch = np.random.randn(batch_size, 768).astype(np.float32)
            batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            metadata = [{"id": j, "small": True} for j in range(i, i + batch_size)]
            store.add(batch, metadata)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Memory usage should be reasonable
        # 100k vectors * 768 dims * 4 bytes = ~300MB for vectors alone
        # With metadata and overhead, should be < 500MB total
        assert memory_used < 500, f"Used {memory_used:.1f}MB for 100k vectors"
        
        print(f"Memory usage: {memory_used:.1f}MB for {n_vectors} vectors")


@pytest.mark.integration
class TestVectorStoreComparison:
    """Compare different vector store implementations."""
    
    def test_consistency_across_stores(self):
        """Test that different stores give consistent results."""
        # Create test data
        embeddings = np.random.randn(50, 256).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metadata = [{"id": i} for i in range(50)]
        
        # Create stores
        memory_store = InMemoryVectorStore(dimension=256)
        faiss_store = FAISSVectorStore(dimension=256, index_type="flat")
        
        # Add same data to both
        memory_store.add(embeddings, metadata)
        faiss_store.add(embeddings, metadata)
        
        # Search with same query
        query = embeddings[0]
        
        memory_results = memory_store.search(query, k=10)
        faiss_results = faiss_store.search(query, k=10)
        
        # Results should be very similar
        assert len(memory_results) == len(faiss_results)
        
        for i in range(10):
            # IDs should match
            assert memory_results[i][1]["id"] == faiss_results[i][1]["id"]
            # Scores should be very close
            assert abs(memory_results[i][0] - faiss_results[i][0]) < 0.001