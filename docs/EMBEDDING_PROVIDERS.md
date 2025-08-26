# Embedding Providers Documentation

ci_fixer_bot uses embeddings for intelligent semantic deduplication of issues. This document describes all available embedding providers and how to configure them.

## Table of Contents
- [Overview](#overview)
- [Provider Comparison](#provider-comparison)
- [Configuration](#configuration)
- [Provider Details](#provider-details)
  - [Local Provider](#local-provider-default)
  - [LM Studio Provider](#lm-studio-provider)
  - [Ollama Provider](#ollama-provider)
  - [OpenAI Provider](#openai-provider)
  - [Mock Provider](#mock-provider-testing-only)
- [Auto-Calibration](#auto-calibration)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

Embedding providers convert text into high-dimensional vectors that capture semantic meaning. This allows ci_fixer_bot to:

- **Detect semantic duplicates**: Issues describing the same problem differently
- **Group related failures**: Cluster similar CI failures together
- **Intelligent matching**: Understand context beyond simple string matching

## Provider Comparison

| Provider | Quality | Speed | Cost | Privacy | Setup Complexity | Best For |
|----------|---------|--------|------|---------|------------------|----------|
| **Local** (default) | Good | Fast | Free | ✅ Full | Easy | Most users, CI/CD |
| **LM Studio** | Excellent | Fast | Free | ✅ Full | Medium | Power users with GPU |
| **Ollama** | Good | Fast | Free | ✅ Full | Easy | Docker users |
| **OpenAI** | Best | Fast | Paid | ❌ Cloud | Easy | Production at scale |
| **Mock** | N/A | Instant | Free | ✅ Full | None | Testing only |

### Embedding Dimensions by Provider

- **Local**: 768 dimensions (all-mpnet-base-v2)
- **LM Studio**: Varies by model (384-4096)
- **Ollama**: 768 (nomic-embed-text) or 1024 (mxbai-embed-large)
- **OpenAI**: 1536 (small) or 3072 (large)

## Configuration

### Basic Configuration

In your `.ci_fixer_bot.yml`:

```yaml
deduplication:
  enabled: true
  use_embeddings: true
  embedding:
    provider: local  # or lm-studio, ollama, openai, mock
    similarity_threshold: 0.80
    auto_calibrate: false
    calibration_strategy: balanced
```

### Provider-Specific Configuration

```yaml
# Local Provider (default)
embedding:
  provider: local
  model: sentence-transformers/all-mpnet-base-v2

# LM Studio
embedding:
  provider: lm-studio
  lm_studio_url: http://localhost:1234/v1
  model: nomic-ai/nomic-embed-text-v1.5-GGUF

# Ollama
embedding:
  provider: ollama
  ollama_url: http://localhost:11434
  ollama_model: nomic-embed-text

# OpenAI
embedding:
  provider: openai
  openai_api_key: ${OPENAI_API_KEY}
  openai_model: text-embedding-3-small
```

## Provider Details

### Local Provider (Default)

The local provider uses sentence-transformers to run embeddings entirely on your machine.

**Installation:**
```bash
pip install sentence-transformers
```

**Models:**
- `sentence-transformers/all-mpnet-base-v2` (default, 420MB)
- `sentence-transformers/all-MiniLM-L6-v2` (faster, 80MB)
- Any Hugging Face sentence-transformer model

**Pros:**
- Completely private and offline
- No API keys required
- Good quality embeddings
- Automatic model caching

**Cons:**
- Requires downloading models (~400MB)
- CPU-only by default (GPU requires PyTorch with CUDA)

**Example:**
```yaml
embedding:
  provider: local
  model: sentence-transformers/all-mpnet-base-v2
```

### LM Studio Provider

LM Studio allows running any GGUF embedding model locally with GPU acceleration.

**Installation:**
1. Download LM Studio: https://lmstudio.ai/
2. Load an embedding model (e.g., nomic-embed-text)
3. Start the local server (Tools → Local Server)

**Recommended Models:**
- `nomic-ai/nomic-embed-text-v1.5-GGUF` (excellent quality)
- `BAAI/bge-small-en-v1.5-GGUF` (fast)
- `BAAI/bge-large-en-v1.5-GGUF` (best quality)

**Pros:**
- GPU acceleration (Metal, CUDA, ROCm)
- Wide model selection
- OpenAI-compatible API
- Visual model management

**Cons:**
- Requires LM Studio installation
- Manual model loading
- Desktop application required

**Example:**
```yaml
embedding:
  provider: lm-studio
  lm_studio_url: http://localhost:1234/v1
  model: nomic-ai/nomic-embed-text-v1.5-GGUF
```

### Ollama Provider

Ollama provides simple model management for local embeddings.

**Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text
```

**Available Models:**
- `nomic-embed-text`: 768-dim, excellent quality (recommended)
- `mxbai-embed-large`: 1024-dim, state-of-the-art
- `all-minilm`: 384-dim, fast and light

**Pros:**
- Simple CLI interface
- Automatic model management
- Docker-friendly
- GPU acceleration when available

**Cons:**
- Sequential batch processing
- Limited model selection
- Requires Ollama daemon

**Example:**
```yaml
embedding:
  provider: ollama
  ollama_url: http://localhost:11434
  ollama_model: nomic-embed-text
```

**Docker Compose:**
```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

### OpenAI Provider

OpenAI provides state-of-the-art embeddings via API.

**Setup:**
1. Get API key: https://platform.openai.com/
2. Set environment variable: `export OPENAI_API_KEY=sk-...`

**Models:**
- `text-embedding-3-small`: 1536-dim, $0.02/1M tokens (default)
- `text-embedding-3-large`: 3072-dim, $0.13/1M tokens
- `text-embedding-ada-002`: 1536-dim, legacy

**Pros:**
- Best quality embeddings
- No local resources needed
- Automatic scaling
- Batch API support

**Cons:**
- Requires API key
- Costs money (though very cheap)
- Data leaves your infrastructure
- Internet connection required

**Example:**
```yaml
embedding:
  provider: openai
  openai_api_key: ${OPENAI_API_KEY}
  openai_model: text-embedding-3-small
```

**Cost Estimation:**
- 1,000 issues ≈ 50,000 tokens ≈ $0.001
- 10,000 issues ≈ 500,000 tokens ≈ $0.01

### Mock Provider (Testing Only)

The mock provider generates deterministic fake embeddings for testing.

**Use Cases:**
- Unit tests
- CI/CD pipelines
- Development without dependencies

**Example:**
```yaml
embedding:
  provider: mock
  embedding_dim: 768  # Match your production provider
```

## Auto-Calibration

ci_fixer_bot can automatically determine the optimal similarity threshold for your repository:

```bash
# Auto-calibrate with balanced strategy
ci_fixer_bot https://github.com/owner/repo --auto-calibrate

# Conservative (high precision)
ci_fixer_bot https://github.com/owner/repo --auto-calibrate --calibration-strategy conservative

# Aggressive (high recall)
ci_fixer_bot https://github.com/owner/repo --auto-calibrate --calibration-strategy aggressive
```

**Strategies:**
- **Conservative**: Minimize false positives (threshold ~0.85)
- **Balanced**: Optimize F1 score (threshold ~0.80)
- **Aggressive**: Minimize false negatives (threshold ~0.75)

## Performance Considerations

### Caching

Embeddings are cached to avoid recomputation:
```yaml
embedding:
  cache_embeddings: true
  cache_path: .ci_fixer_bot_cache
```

### Vector Store

For repositories with many issues:
```yaml
embedding:
  vector_store_type: memory  # Default, fast for < 10,000 issues
  # vector_store_type: faiss  # For > 10,000 issues (requires faiss-cpu)
```

### Batch Processing

Batch sizes by provider:
- **Local**: Unlimited (processes all at once)
- **LM Studio**: Unlimited
- **Ollama**: Sequential (processes one at a time)
- **OpenAI**: 100 texts per request

## Troubleshooting

### Local Provider Issues

**Error: "sentence-transformers is required"**
```bash
pip install sentence-transformers
```

**Slow performance:**
```bash
# Install with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### LM Studio Issues

**Error: "Cannot connect to LM Studio"**
1. Ensure LM Studio is running
2. Start local server: Tools → Local Server → Start
3. Check URL matches config (default: http://localhost:1234/v1)

### Ollama Issues

**Error: "Model not found"**
```bash
# Pull the model
ollama pull nomic-embed-text

# List available models
ollama list
```

**Docker networking:**
```yaml
# Use host network mode
embedding:
  ollama_url: http://host.docker.internal:11434  # For Docker Desktop
  # ollama_url: http://172.17.0.1:11434  # For Linux
```

### OpenAI Issues

**Error: "Invalid API key"**
```bash
# Check key is set
echo $OPENAI_API_KEY

# Test API access
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Rate limiting:**
The provider automatically retries with exponential backoff.

### General Debugging

**Enable verbose logging:**
```bash
ci_fixer_bot https://github.com/owner/repo --verbose
```

**Test embedding provider:**
```python
from ci_fixer_bot.embedding_providers import create_embedding_provider
from ci_fixer_bot.config import load_config

config = load_config(".ci_fixer_bot.yml")
provider = create_embedding_provider(config)

# Test embedding
embedding = provider.embed_text("test")
print(f"Embedding shape: {embedding.shape}")
print(f"Provider available: {provider.is_available()}")
```

## Recommendations by Use Case

### CI/CD Pipeline
Use **Local** or **Mock** provider:
```yaml
embedding:
  provider: local  # or mock for faster tests
  model: sentence-transformers/all-MiniLM-L6-v2  # Smaller, faster model
```

### Development Machine with GPU
Use **LM Studio**:
```yaml
embedding:
  provider: lm-studio
  lm_studio_url: http://localhost:1234/v1
```

### Docker/Kubernetes
Use **Ollama**:
```yaml
embedding:
  provider: ollama
  ollama_url: http://ollama-service:11434
```

### Production at Scale
Use **OpenAI**:
```yaml
embedding:
  provider: openai
  openai_api_key: ${OPENAI_API_KEY}
  openai_model: text-embedding-3-small
```

### Testing
Use **Mock**:
```yaml
embedding:
  provider: mock
  embedding_dim: 768
```