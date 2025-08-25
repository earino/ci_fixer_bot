"""
Integration tests for LocalEmbeddingProvider with real models.

Note: These tests require sentence-transformers to be installed and will
download a small model (~80MB) on first run. They are marked as integration
tests and can be skipped in CI if needed.
"""

import numpy as np
import pytest

# Skip all tests in this file if sentence-transformers not available
pytest.importorskip("sentence_transformers")

from ci_fixer_bot.embedding_providers_local import LocalEmbeddingProvider
from ci_fixer_bot.config import Config
from ci_fixer_bot.embedding_providers import create_embedding_provider, validate_embedding_provider


@pytest.mark.integration
class TestLocalEmbeddingProviderIntegration:
    """Integration tests with real sentence-transformers models."""
    
    @pytest.fixture
    def provider(self):
        """Create a provider with a small, fast model for testing."""
        # Use all-MiniLM-L6-v2 - it's small (80MB) and fast
        return LocalEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",  # Force CPU for consistent testing
            show_progress_bar=False  # Disable for cleaner test output
        )
    
    def test_real_model_loading(self, provider):
        """Test that a real model can be loaded."""
        assert provider._model is None
        
        # First embedding triggers loading
        embedding = provider.embed_text("test")
        
        assert provider._model is not None
        assert provider.is_loaded
        assert provider.get_embedding_dim() == 384  # all-MiniLM-L6-v2 dimension
    
    def test_real_single_embedding(self, provider):
        """Test embedding a single text with real model."""
        text = "CI failure in authentication module"
        embedding = provider.embed_text(text)
        
        # Check properties
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
        assert embedding.dtype == np.float32
        
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, rtol=1e-5)
        
        # Should be deterministic
        embedding2 = provider.embed_text(text)
        np.testing.assert_array_almost_equal(embedding, embedding2)
    
    def test_real_batch_embedding(self, provider):
        """Test batch embedding with real model."""
        texts = [
            "Authentication test failures",
            "Database connection timeout",
            "Linting errors in main.py",
            "Memory leak detected",
            "API rate limit exceeded"
        ]
        
        embeddings = provider.embed_batch(texts)
        
        # Check shape
        assert embeddings.shape == (5, 384)
        
        # All should be normalized
        for i in range(5):
            norm = np.linalg.norm(embeddings[i])
            assert np.isclose(norm, 1.0, rtol=1e-5)
        
        # Different texts should have different embeddings
        for i in range(5):
            for j in range(i + 1, 5):
                similarity = np.dot(embeddings[i], embeddings[j])
                # Similarity should be less than 0.99 (not identical)
                assert similarity < 0.99
    
    def test_semantic_similarity(self, provider):
        """Test that semantically similar texts have high similarity."""
        # Similar texts
        text1 = "Authentication tests are failing"
        text2 = "Auth test failures detected"
        text3 = "Database connection dropped unexpectedly"
        
        emb1 = provider.embed_text(text1)
        emb2 = provider.embed_text(text2)
        emb3 = provider.embed_text(text3)
        
        # Similar texts should have high similarity
        sim_12 = np.dot(emb1, emb2)
        # Different texts should have lower similarity
        sim_13 = np.dot(emb1, emb3)
        sim_23 = np.dot(emb2, emb3)
        
        # Auth-related texts should be more similar to each other
        assert sim_12 > sim_13
        assert sim_12 > sim_23
        
        # Sanity check - similarity should be in valid range
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0
        assert -1.0 <= sim_23 <= 1.0
    
    def test_edge_cases(self, provider):
        """Test edge cases with real model."""
        # Empty string
        empty_emb = provider.embed_text("")
        assert empty_emb.shape == (384,)
        
        # Very long text (should be truncated by model)
        long_text = "test " * 1000
        long_emb = provider.embed_text(long_text)
        assert long_emb.shape == (384,)
        
        # Special characters
        special_emb = provider.embed_text("🚀 CI/CD #fail @github $test")
        assert special_emb.shape == (384,)
        
        # Single word
        word_emb = provider.embed_text("failure")
        assert word_emb.shape == (384,)
    
    def test_model_unloading(self, provider):
        """Test model can be loaded and unloaded."""
        # Load model
        provider.embed_text("test")
        assert provider.is_loaded
        
        # Unload
        provider.unload_model()
        assert not provider.is_loaded
        assert provider._model is None
        
        # Can load again
        provider.embed_text("test again")
        assert provider.is_loaded
    
    def test_factory_creation(self):
        """Test creating LocalEmbeddingProvider via factory."""
        config = Config()
        config.deduplication.use_embeddings = True
        config.deduplication.embedding.provider = "local"
        config.deduplication.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # This will fail with ImportError if sentence-transformers not installed
        # But since we skip this file without it, we're good
        provider = create_embedding_provider(config)
        
        assert isinstance(provider, LocalEmbeddingProvider)
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_provider_validation(self, provider):
        """Test that the real provider passes validation."""
        assert validate_embedding_provider(provider) is True
    
    @pytest.mark.slow
    def test_large_batch_performance(self, provider):
        """Test performance with large batch."""
        # Generate 100 different texts
        texts = [f"Test document number {i} with some content" for i in range(100)]
        
        embeddings = provider.embed_batch(texts)
        
        assert embeddings.shape == (100, 384)
        
        # All should be different
        for i in range(10):  # Check first 10 for speed
            for j in range(i + 1, 10):
                similarity = np.dot(embeddings[i], embeddings[j])
                assert similarity < 0.99  # Not identical


@pytest.mark.integration 
class TestDifferentModels:
    """Test with different recommended models."""
    
    @pytest.mark.parametrize("model_name,expected_dim", [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        # Uncomment to test more models (will download them):
        # ("sentence-transformers/all-mpnet-base-v2", 768),
        # ("sentence-transformers/all-distilroberta-v1", 768),
    ])
    def test_different_models(self, model_name, expected_dim):
        """Test that different models work correctly."""
        provider = LocalEmbeddingProvider(
            model_name=model_name,
            show_progress_bar=False
        )
        
        embedding = provider.embed_text("test")
        
        assert embedding.shape == (expected_dim,)
        assert provider.get_embedding_dim() == expected_dim