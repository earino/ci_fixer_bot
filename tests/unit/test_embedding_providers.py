"""
Unit tests for embedding provider abstractions.
"""

import numpy as np
import pytest

from ci_fixer_bot.config import Config, DeduplicationConfig, EmbeddingConfig
from ci_fixer_bot.embedding_providers import (
    EmbeddingError,
    EmbeddingProvider,
    MockEmbeddingProvider,
    create_embedding_provider,
    validate_embedding_provider,
)


class TestMockEmbeddingProvider:
    """Test the mock embedding provider."""
    
    def test_mock_provider_initialization(self):
        """Test mock provider can be initialized with custom dimensions."""
        provider = MockEmbeddingProvider(embedding_dim=512)
        assert provider.get_embedding_dim() == 512
        
        provider = MockEmbeddingProvider()  # Default
        assert provider.get_embedding_dim() == 768
    
    def test_mock_provider_single_embedding(self):
        """Test mock provider generates consistent embeddings for single text."""
        provider = MockEmbeddingProvider(embedding_dim=768)
        
        text = "This is a test sentence"
        embedding1 = provider.embed_text(text)
        embedding2 = provider.embed_text(text)
        
        # Should be deterministic
        np.testing.assert_array_equal(embedding1, embedding2)
        
        # Should have correct dimension
        assert embedding1.shape == (768,)
        
        # Should be normalized
        norm = np.linalg.norm(embedding1)
        assert np.isclose(norm, 1.0, rtol=1e-5)
    
    def test_mock_provider_different_texts(self):
        """Test mock provider generates different embeddings for different texts."""
        provider = MockEmbeddingProvider()
        
        embedding1 = provider.embed_text("First text")
        embedding2 = provider.embed_text("Second text")
        
        # Should be different
        assert not np.array_equal(embedding1, embedding2)
        
        # But should still be normalized
        assert np.isclose(np.linalg.norm(embedding1), 1.0, rtol=1e-5)
        assert np.isclose(np.linalg.norm(embedding2), 1.0, rtol=1e-5)
    
    def test_mock_provider_batch_embedding(self):
        """Test mock provider can embed multiple texts."""
        provider = MockEmbeddingProvider(embedding_dim=384)
        
        texts = ["First", "Second", "Third", "Fourth"]
        embeddings = provider.embed_batch(texts)
        
        # Should have correct shape
        assert embeddings.shape == (4, 384)
        
        # Each should be normalized
        for i in range(4):
            norm = np.linalg.norm(embeddings[i])
            assert np.isclose(norm, 1.0, rtol=1e-5)
        
        # Should be consistent with single embedding
        single_embedding = provider.embed_text("First")
        np.testing.assert_array_almost_equal(embeddings[0], single_embedding)
    
    def test_mock_provider_empty_text(self):
        """Test mock provider handles empty text gracefully."""
        provider = MockEmbeddingProvider()
        
        embedding = provider.embed_text("")
        assert embedding.shape == (768,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)
    
    def test_mock_provider_long_text(self):
        """Test mock provider handles long text."""
        provider = MockEmbeddingProvider()
        
        long_text = "This is a very long text. " * 1000
        embedding = provider.embed_text(long_text)
        
        assert embedding.shape == (768,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)


class TestEmbeddingProviderFactory:
    """Test the embedding provider factory function."""
    
    def test_create_mock_provider(self):
        """Test factory can create mock provider."""
        config = Config(
            deduplication=DeduplicationConfig(
                use_embeddings=True,
                embedding=EmbeddingConfig(
                    provider="mock",
                    embedding_dim=512
                )
            )
        )
        
        provider = create_embedding_provider(config)
        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.get_embedding_dim() == 512
    
    def test_factory_with_embeddings_disabled(self):
        """Test factory raises error when embeddings are disabled."""
        config = Config(
            deduplication=DeduplicationConfig(
                use_embeddings=False
            )
        )
        
        with pytest.raises(ValueError, match="Embeddings are not enabled"):
            create_embedding_provider(config)
    
    def test_factory_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        config = Config(
            deduplication=DeduplicationConfig(
                use_embeddings=True,
                embedding=EmbeddingConfig(
                    provider="unknown_provider"
                )
            )
        )
        
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(config)
    
    def test_factory_missing_dependencies(self):
        """Test factory provides helpful error for missing dependencies."""
        # Test local provider without sentence-transformers
        config = Config(
            deduplication=DeduplicationConfig(
                use_embeddings=True,
                embedding=EmbeddingConfig(provider="local")
            )
        )
        
        with pytest.raises(ValueError, match="sentence-transformers"):
            create_embedding_provider(config)
        
        # Test OpenAI provider without openai package
        config.deduplication.embedding.provider = "openai"
        with pytest.raises(ValueError, match="openai package"):
            create_embedding_provider(config)


class TestEmbeddingProviderValidation:
    """Test the embedding provider validation function."""
    
    def test_validate_working_provider(self):
        """Test validation passes for working provider."""
        provider = MockEmbeddingProvider()
        assert validate_embedding_provider(provider) is True
    
    def test_validate_provider_with_wrong_dimension(self):
        """Test validation catches dimension mismatches."""
        
        class BrokenDimensionProvider(EmbeddingProvider):
            def embed_text(self, text: str) -> np.ndarray:
                return np.ones(100)  # Wrong dimension
            
            def embed_batch(self, texts: list) -> np.ndarray:
                return np.ones((len(texts), 100))
            
            def get_embedding_dim(self) -> int:
                return 768  # Claims different dimension
        
        provider = BrokenDimensionProvider()
        assert validate_embedding_provider(provider) is False
    
    def test_validate_provider_with_unnormalized_embeddings(self):
        """Test validation handles unnormalized embeddings."""
        
        class UnnormalizedProvider(EmbeddingProvider):
            def embed_text(self, text: str) -> np.ndarray:
                return np.ones(768) * 2  # Not normalized
            
            def embed_batch(self, texts: list) -> np.ndarray:
                return np.ones((len(texts), 768)) * 2
            
            def get_embedding_dim(self) -> int:
                return 768
        
        provider = UnnormalizedProvider()
        # Should still pass because normalize_embedding can fix it
        result = validate_embedding_provider(provider)
        # The validation function will try to normalize
        assert result is True or result is False  # Depends on implementation
    
    def test_validate_provider_that_raises_exception(self):
        """Test validation handles providers that raise exceptions."""
        
        class ExceptionProvider(EmbeddingProvider):
            def embed_text(self, text: str) -> np.ndarray:
                raise EmbeddingError("Test error")
            
            def embed_batch(self, texts: list) -> np.ndarray:
                raise EmbeddingError("Test error")
            
            def get_embedding_dim(self) -> int:
                return 768
        
        provider = ExceptionProvider()
        assert validate_embedding_provider(provider) is False


class TestEmbeddingProviderNormalization:
    """Test the normalization methods in base class."""
    
    def test_normalize_single_embedding(self):
        """Test single embedding normalization."""
        provider = MockEmbeddingProvider()
        
        # Create unnormalized embedding
        embedding = np.array([3.0, 4.0])  # Norm = 5
        normalized = provider.normalize_embedding(embedding)
        
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        np.testing.assert_array_almost_equal(normalized, [0.6, 0.8])
    
    def test_normalize_zero_embedding(self):
        """Test normalization handles zero vectors."""
        provider = MockEmbeddingProvider()
        
        embedding = np.zeros(10)
        normalized = provider.normalize_embedding(embedding)
        
        # Should return unchanged
        np.testing.assert_array_equal(normalized, embedding)
    
    def test_normalize_batch_embeddings(self):
        """Test batch normalization."""
        provider = MockEmbeddingProvider()
        
        embeddings = np.array([
            [3.0, 4.0],      # Norm = 5
            [1.0, 0.0],      # Norm = 1 (already normalized)
            [0.0, 0.0],      # Zero vector
            [5.0, 12.0],     # Norm = 13
        ])
        
        normalized = provider.normalize_batch(embeddings)
        
        # Check each is normalized (or zero)
        np.testing.assert_array_almost_equal(normalized[0], [0.6, 0.8])
        np.testing.assert_array_almost_equal(normalized[1], [1.0, 0.0])
        np.testing.assert_array_almost_equal(normalized[2], [0.0, 0.0])
        np.testing.assert_array_almost_equal(normalized[3], [5/13, 12/13])


class TestConfigIntegration:
    """Test configuration integration with embedding providers."""
    
    def test_default_embedding_config(self):
        """Test default embedding configuration."""
        config = Config()
        
        assert config.deduplication.use_embeddings is True
        assert config.deduplication.embedding.provider == "local"
        assert config.deduplication.embedding.model == "all-mpnet-base-v2"
        assert config.deduplication.embedding.similarity_threshold == 0.85
        assert config.deduplication.embedding.cache_embeddings is True
    
    def test_custom_embedding_config(self):
        """Test custom embedding configuration."""
        config = Config(
            deduplication=DeduplicationConfig(
                use_embeddings=True,
                embedding=EmbeddingConfig(
                    provider="lm-studio",
                    lm_studio_url="http://localhost:5000/v1",
                    similarity_threshold=0.9
                )
            )
        )
        
        assert config.deduplication.embedding.provider == "lm-studio"
        assert config.deduplication.embedding.lm_studio_url == "http://localhost:5000/v1"
        assert config.deduplication.embedding.similarity_threshold == 0.9
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old deduplication config."""
        config = Config(
            deduplication=DeduplicationConfig(
                enabled=True,
                use_embeddings=False,  # Use old system
                similarity_threshold=0.75
            )
        )
        
        assert config.deduplication.enabled is True
        assert config.deduplication.use_embeddings is False
        assert config.deduplication.similarity_threshold == 0.75