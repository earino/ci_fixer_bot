"""
Unit tests for LocalEmbeddingProvider.

These tests mock sentence-transformers to test the provider logic
without requiring the actual library to be installed.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys

from ci_fixer_bot.embedding_providers import EmbeddingError
from ci_fixer_bot.embedding_providers_local import (
    LocalEmbeddingProvider,
    get_model_info,
    RECOMMENDED_MODELS
)


class TestLocalEmbeddingProviderInit:
    """Test LocalEmbeddingProvider initialization."""
    
    def test_default_initialization(self):
        """Test provider initializes with default settings."""
        provider = LocalEmbeddingProvider()
        
        assert provider.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert provider.device is None
        assert provider.normalize_embeddings is True
        assert provider.show_progress_bar is True
        assert provider.batch_size == 32
        assert provider._model is None  # Lazy loading
        assert provider.is_loaded is False
    
    def test_custom_initialization(self):
        """Test provider initializes with custom settings."""
        provider = LocalEmbeddingProvider(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            normalize_embeddings=False,
            show_progress_bar=False,
            batch_size=64
        )
        
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.device == "cpu"
        assert provider.normalize_embeddings is False
        assert provider.show_progress_bar is False
        assert provider.batch_size == 64


class TestLocalEmbeddingProviderWithMock:
    """Test LocalEmbeddingProvider with mocked sentence-transformers."""
    
    def test_lazy_loading(self):
        """Test model is loaded lazily on first use."""
        # Create provider
        provider = LocalEmbeddingProvider()
        
        # Model should not be loaded yet
        assert provider._model is None
        assert provider.is_loaded is False
        
        # Mock sentence_transformers for loading
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768)
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            # First embed should trigger loading
            embedding = provider.embed_text("test")
            
            # Now model should be loaded
            assert provider._model is not None
            assert provider.is_loaded is True
            mock_st.SentenceTransformer.assert_called_once_with(
                "sentence-transformers/all-mpnet-base-v2",
                device=None
            )
            
            # Model should be in eval mode
            mock_model.eval.assert_called_once()
    
    def test_single_text_embedding(self):
        """Test embedding a single text."""
        provider = LocalEmbeddingProvider()
        
        # Setup mock
        mock_embedding = np.random.randn(768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embedding
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            result = provider.embed_text("test text")
            
            # Check encode was called correctly (look for the actual text in any call)
            encode_calls = mock_model.encode.call_args_list
            assert any("test text" in str(call) for call in encode_calls)
            
            # Check result
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
    
    def test_batch_embedding(self):
        """Test embedding multiple texts."""
        provider = LocalEmbeddingProvider()
        
        # Setup mock
        batch_size = 5
        mock_embeddings = np.random.randn(batch_size, 768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            texts = ["text1", "text2", "text3", "text4", "text5"]
            result = provider.embed_batch(texts)
            
            # Check encode was called with the texts
            encode_calls = mock_model.encode.call_args_list
            assert any(texts == call[0][0] for call in encode_calls)
            
            # Check result
            assert isinstance(result, np.ndarray)
            assert result.shape == (5, 768)
            assert result.dtype == np.float32
    
    def test_batch_embedding_with_progress(self):
        """Test batch embedding shows progress for large batches."""
        provider = LocalEmbeddingProvider(show_progress_bar=True)
        
        # Setup mock
        batch_size = 100
        mock_embeddings = np.random.randn(batch_size, 768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            texts = ["text"] * 100  # Large batch
            result = provider.embed_batch(texts)
            
            # Check that encode was called with show_progress_bar=True
            encode_calls = mock_model.encode.call_args_list
            # Find the actual batch encoding call (not the dummy one)
            batch_call = [c for c in encode_calls if len(c[0][0]) == 100]
            if batch_call:
                assert batch_call[0][1].get('show_progress_bar', False) == True
    
    def test_get_embedding_dim(self):
        """Test getting embedding dimension."""
        provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(384)  # Different dimension
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            dim = provider.get_embedding_dim()
            
            assert dim == 384
            assert provider._embedding_dim == 384
    
    def test_cpu_optimization(self):
        """Test CPU optimization settings."""
        provider = LocalEmbeddingProvider(device='cpu')
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768)
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.set_num_threads = MagicMock()
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st, 'torch': mock_torch}):
            with patch('os.cpu_count', return_value=8):
                provider.embed_text("test")
                
                # Should set thread count for CPU
                mock_torch.set_num_threads.assert_called_once_with(8)
    
    def test_model_unloading(self):
        """Test model can be unloaded from memory."""
        provider = LocalEmbeddingProvider()
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768)
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            # Load model
            provider.embed_text("test")
            assert provider.is_loaded is True
            
            # Unload model
            provider.unload_model()
            assert provider._model is None
            assert provider.is_loaded is False
    
    def test_missing_sentence_transformers(self):
        """Test helpful error when sentence-transformers not installed."""
        provider = LocalEmbeddingProvider()
        
        # Don't mock sentence_transformers, so import will fail
        with pytest.raises(EmbeddingError, match="sentence-transformers is required"):
            provider.embed_text("test")
    
    def test_model_loading_failure(self):
        """Test error handling when model fails to load."""
        provider = LocalEmbeddingProvider(model_name="non-existent-model")
        
        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            with pytest.raises(EmbeddingError, match="Failed to load model"):
                provider.embed_text("test")
    
    def test_embedding_failure(self):
        """Test error handling when embedding fails."""
        provider = LocalEmbeddingProvider()
        
        mock_model = MagicMock()
        # First call succeeds (for dimension check), subsequent calls fail
        mock_model.encode.side_effect = [np.ones(768), Exception("Encoding failed")]
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            with pytest.raises(EmbeddingError, match="Failed to embed text"):
                provider.embed_text("test")
    
    def test_empty_batch(self):
        """Test handling empty batch."""
        provider = LocalEmbeddingProvider()
        
        # Empty batch shouldn't trigger model loading
        result = provider.embed_batch([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert provider._model is None  # Shouldn't be loaded


class TestModelInfo:
    """Test model information utilities."""
    
    def test_get_known_model_info(self):
        """Test getting info for known models."""
        info = get_model_info("all-mpnet-base-v2")
        assert info["dimension"] == 768
        assert info["quality"] == "excellent"
        
        info = get_model_info("sentence-transformers/all-MiniLM-L6-v2")
        assert info["dimension"] == 384
        assert info["speed"] == "very fast"
    
    def test_get_unknown_model_info(self):
        """Test getting info for unknown models."""
        info = get_model_info("custom-model")
        assert info["description"] == "Custom model"
        assert info["dimension"] == "unknown"
    
    def test_recommended_models(self):
        """Test recommended models are properly defined."""
        assert "all-mpnet-base-v2" in RECOMMENDED_MODELS
        assert "all-MiniLM-L6-v2" in RECOMMENDED_MODELS
        
        for model_name, info in RECOMMENDED_MODELS.items():
            assert "description" in info
            assert "dimension" in info
            assert "speed" in info
            assert "quality" in info