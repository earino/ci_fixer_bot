"""
Tests for additional embedding providers (LM Studio, Ollama, OpenAI).

These tests use mocking to avoid requiring actual services to be running.
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from src.ci_fixer_bot.embedding_providers import EmbeddingError


class TestLMStudioProvider(unittest.TestCase):
    """Test LM Studio embedding provider."""
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialization_success(self, mock_post, mock_get):
        """Test successful initialization."""
        from src.ci_fixer_bot.embedding_providers_lmstudio import LMStudioEmbeddingProvider
        
        # Mock successful connection test
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [{"id": "test-model"}]
        }
        
        provider = LMStudioEmbeddingProvider(url="http://localhost:1234/v1")
        
        self.assertEqual(provider.url, "http://localhost:1234/v1")
        self.assertEqual(provider.model, "test-model")
        mock_get.assert_called()
    
    @patch('requests.post')
    def test_embed_text(self, mock_post):
        """Test embedding single text."""
        from src.ci_fixer_bot.embedding_providers_lmstudio import LMStudioEmbeddingProvider
        
        # Mock the connection test
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
            provider = LMStudioEmbeddingProvider()
        
        # Mock embedding response
        mock_embedding = np.random.rand(768).tolist()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": mock_embedding, "index": 0}]
        }
        
        embedding = provider.embed_text("test text")
        
        self.assertEqual(embedding.shape, (768,))
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_embed_batch(self, mock_post):
        """Test batch embedding."""
        from src.ci_fixer_bot.embedding_providers_lmstudio import LMStudioEmbeddingProvider
        
        # Mock the connection test
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
            provider = LMStudioEmbeddingProvider()
        
        # Mock batch response
        mock_embeddings = [np.random.rand(768).tolist() for _ in range(3)]
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "data": [
                {"embedding": mock_embeddings[i], "index": i}
                for i in range(3)
            ]
        }
        
        texts = ["text1", "text2", "text3"]
        embeddings = provider.embed_batch(texts)
        
        self.assertEqual(embeddings.shape, (3, 768))
        # Check all are normalized
        for i in range(3):
            self.assertAlmostEqual(np.linalg.norm(embeddings[i]), 1.0, places=5)
    
    @patch('requests.get')
    def test_connection_failure(self, mock_get):
        """Test handling of connection failure."""
        from src.ci_fixer_bot.embedding_providers_lmstudio import LMStudioEmbeddingProvider
        
        mock_get.side_effect = Exception("Connection failed")
        
        with self.assertRaises(EmbeddingError):
            LMStudioEmbeddingProvider()


class TestOllamaProvider(unittest.TestCase):
    """Test Ollama embedding provider."""
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialization_model_available(self, mock_post, mock_get):
        """Test initialization when model is available."""
        from src.ci_fixer_bot.embedding_providers_ollama import OllamaEmbeddingProvider
        
        # Mock model list response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "nomic-embed-text"}]
        }
        
        provider = OllamaEmbeddingProvider()
        
        self.assertEqual(provider.model, "nomic-embed-text")
        mock_get.assert_called()
        mock_post.assert_not_called()  # Should not pull if available
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialization_pull_model(self, mock_post, mock_get):
        """Test pulling model when not available."""
        from src.ci_fixer_bot.embedding_providers_ollama import OllamaEmbeddingProvider
        
        # Model not in list
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": []}
        
        # Mock pull success
        mock_post.return_value.status_code = 200
        
        provider = OllamaEmbeddingProvider()
        
        # Should have attempted to pull
        mock_post.assert_called_with(
            "http://localhost:11434/api/pull",
            json={"name": "nomic-embed-text"},
            timeout=300
        )
    
    @patch('requests.post')
    def test_embed_text(self, mock_post):
        """Test embedding single text."""
        from src.ci_fixer_bot.embedding_providers_ollama import OllamaEmbeddingProvider
        
        # Mock initialization
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "nomic-embed-text"}]
            }
            provider = OllamaEmbeddingProvider()
        
        # Mock embedding response
        mock_embedding = np.random.rand(768).tolist()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "embedding": mock_embedding
        }
        
        embedding = provider.embed_text("test text")
        
        self.assertEqual(embedding.shape, (768,))
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    @patch('requests.post')
    def test_embed_batch_sequential(self, mock_post):
        """Test batch embedding (sequential processing)."""
        from src.ci_fixer_bot.embedding_providers_ollama import OllamaEmbeddingProvider
        
        # Mock initialization
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "nomic-embed-text"}]
            }
            provider = OllamaEmbeddingProvider()
        
        # Mock individual embedding responses
        mock_embeddings = [np.random.rand(768).tolist() for _ in range(3)]
        mock_post.return_value.status_code = 200
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"embedding": emb})
            for emb in mock_embeddings
        ]
        
        texts = ["text1", "text2", "text3"]
        embeddings = provider.embed_batch(texts)
        
        self.assertEqual(embeddings.shape, (3, 768))
        self.assertEqual(mock_post.call_count, 3)  # Called once per text


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI embedding provider."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.post')
    def test_initialization_with_env_key(self, mock_post):
        """Test initialization with environment variable."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        # Mock API test
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        
        provider = OpenAIEmbeddingProvider()
        
        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.model, "text-embedding-3-small")
    
    def test_initialization_without_key(self):
        """Test initialization fails without API key."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError) as context:
                OpenAIEmbeddingProvider()
            
            self.assertIn("API key required", str(context.exception))
    
    @patch('requests.post')
    def test_embed_text(self, mock_post):
        """Test embedding single text."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        # Initialize with explicit API key
        with patch.object(OpenAIEmbeddingProvider, '_test_api', return_value=True):
            provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        # Mock embedding response
        mock_embedding = np.random.rand(1536).tolist()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": mock_embedding, "index": 0}],
            "usage": {"total_tokens": 5}
        }
        
        embedding = provider.embed_text("test text")
        
        self.assertEqual(embedding.shape, (1536,))
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    @patch('requests.post')
    def test_embed_batch_with_splitting(self, mock_post):
        """Test batch embedding with automatic splitting."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        with patch.object(OpenAIEmbeddingProvider, '_test_api', return_value=True):
            provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        # Create 150 texts (will be split into 2 batches)
        texts = [f"text{i}" for i in range(150)]
        
        # Mock responses for two batches
        batch1_embeddings = [np.random.rand(1536).tolist() for _ in range(100)]
        batch2_embeddings = [np.random.rand(1536).tolist() for _ in range(50)]
        
        mock_post.side_effect = [
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [{"embedding": emb, "index": i} for i, emb in enumerate(batch1_embeddings)]
                }
            ),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [{"embedding": emb, "index": i} for i, emb in enumerate(batch2_embeddings)]
                }
            )
        ]
        
        embeddings = provider.embed_batch(texts)
        
        self.assertEqual(embeddings.shape, (150, 1536))
        self.assertEqual(mock_post.call_count, 2)  # Two batches
    
    @patch('requests.post')
    @patch('time.sleep')
    def test_retry_on_rate_limit(self, mock_sleep, mock_post):
        """Test retry logic on rate limiting."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        with patch.object(OpenAIEmbeddingProvider, '_test_api', return_value=True):
            provider = OpenAIEmbeddingProvider(api_key="test-key", max_retries=3)
        
        # First call rate limited, second succeeds
        mock_embedding = np.random.rand(1536).tolist()
        mock_post.side_effect = [
            MagicMock(status_code=429),  # Rate limited
            MagicMock(
                status_code=200,
                json=lambda: {"data": [{"embedding": mock_embedding, "index": 0}]}
            )
        ]
        
        embedding = provider.embed_text("test")
        
        self.assertEqual(embedding.shape, (1536,))
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()  # Should have waited
    
    def test_get_embedding_dim(self):
        """Test getting embedding dimensions for different models."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        with patch.object(OpenAIEmbeddingProvider, '_test_api', return_value=True):
            # Small model
            provider_small = OpenAIEmbeddingProvider(
                api_key="test", 
                model="text-embedding-3-small"
            )
            self.assertEqual(provider_small.get_embedding_dim(), 1536)
            
            # Large model
            provider_large = OpenAIEmbeddingProvider(
                api_key="test",
                model="text-embedding-3-large"
            )
            self.assertEqual(provider_large.get_embedding_dim(), 3072)
    
    def test_cost_estimation(self):
        """Test cost estimation."""
        from src.ci_fixer_bot.embedding_providers_openai import OpenAIEmbeddingProvider
        
        with patch.object(OpenAIEmbeddingProvider, '_test_api', return_value=True):
            provider = OpenAIEmbeddingProvider(
                api_key="test",
                model="text-embedding-3-small"
            )
            
            # 1 million tokens
            cost = provider.estimate_cost(1_000_000)
            self.assertAlmostEqual(cost, 0.02, places=4)
            
            # 100k tokens
            cost = provider.estimate_cost(100_000)
            self.assertAlmostEqual(cost, 0.002, places=4)


if __name__ == "__main__":
    unittest.main()