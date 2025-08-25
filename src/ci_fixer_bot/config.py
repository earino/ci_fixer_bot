"""
Configuration management for ci_fixer_bot.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = "claude-cli"
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    command: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: int = 4000
    temperature: float = 0.1


class RiskOverride(BaseModel):
    """Risk level overrides for specific file patterns."""
    pattern: str
    risk_level: str  # safe, tactical, strategic


class TeamMapping(BaseModel):
    """Team assignments for file patterns."""
    team: str
    patterns: List[str]


class AutoFixConfig(BaseModel):
    """Configuration for automatic fixes."""
    enabled: bool = True
    types: List[str] = Field(default_factory=lambda: ["trailing_whitespace", "import_sorting"])
    require_approval: bool = False


class NotificationConfig(BaseModel):
    """Configuration for notifications."""
    slack_webhook: Optional[str] = None
    email: Optional[str] = None
    enabled: bool = True


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    provider: str = "local"  # local, lm-studio, ollama, openai, mock
    model: Optional[str] = "all-mpnet-base-v2"  # For local provider
    similarity_threshold: float = 0.85
    cache_embeddings: bool = True
    cache_path: str = ".ci_fixer_bot_cache"
    embedding_dim: Optional[int] = None  # For mock provider
    
    # Provider-specific settings
    lm_studio_url: str = "http://localhost:1234/v1"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"


class DeduplicationConfig(BaseModel):
    """Configuration for issue deduplication."""
    enabled: bool = True
    use_embeddings: bool = True  # Use embeddings-based deduplication
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Legacy settings for backward compatibility
    similarity_threshold: float = 0.8  # Minimum similarity score to consider a duplicate (0.0-1.0)
    exact_match_threshold: float = 0.95  # Threshold for exact matches (0.0-1.0)
    update_existing: bool = True  # Whether to add comments to existing issues when duplicates are found
    search_limit: int = 20  # Maximum number of existing issues to check for duplicates


class Config(BaseModel):
    """Main configuration class for ci_fixer_bot."""
    
    # Default settings
    analyze_runs: int = 5
    focus: str = "all"  # safe, tactical, strategic, all
    min_priority: int = 1
    
    # LLM configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Risk assessment
    risk_overrides: List[RiskOverride] = Field(default_factory=list)
    
    # Team mappings
    team_mappings: List[TeamMapping] = Field(default_factory=list)
    
    # Auto-fix settings
    auto_fix: AutoFixConfig = Field(default_factory=AutoFixConfig)
    
    # Notifications
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    
    # Issue deduplication
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    
    # GitHub settings
    github_token: Optional[str] = None


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file and environment variables.
    
    Priority order:
    1. Provided config_path
    2. .ci_fixer_bot.yml in current directory
    3. .ci_fixer_bot.yml in home directory
    4. Default configuration
    """
    config_data = {}
    
    # Try to find config file
    if config_path is None:
        # Look for config in current directory
        current_dir_config = Path.cwd() / ".ci_fixer_bot.yml"
        home_dir_config = Path.home() / ".ci_fixer_bot.yml"
        
        if current_dir_config.exists():
            config_path = current_dir_config
        elif home_dir_config.exists():
            config_path = home_dir_config
    
    # Load config file if found
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Override with environment variables
    env_overrides = {
        "github_token": os.getenv("GITHUB_TOKEN"),
        "llm.api_key": os.getenv("OPENAI_API_KEY") or os.getenv("CLAUDE_API_KEY"),
        "notifications.slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
        "notifications.email": os.getenv("NOTIFICATION_EMAIL"),
        "deduplication.embedding.provider": os.getenv("EMBEDDING_PROVIDER"),
        "deduplication.embedding.openai_api_key": os.getenv("OPENAI_API_KEY"),
        "deduplication.embedding.lm_studio_url": os.getenv("LM_STUDIO_URL"),
        "deduplication.embedding.ollama_url": os.getenv("OLLAMA_URL"),
    }
    
    # Apply environment overrides
    for key, value in env_overrides.items():
        if value is not None:
            _set_nested_dict(config_data, key, value)
    
    # Create config object
    config = Config(**config_data)
    
    # Validate required settings
    _validate_config(config)
    
    return config


def _set_nested_dict(d: dict, key: str, value: str) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _validate_config(config: Config) -> None:
    """Validate configuration and provide helpful error messages."""
    
    # Check GitHub token
    if not config.github_token:
        # Try to get from git config or gh CLI
        import subprocess
        try:
            result = subprocess.run(
                ["gh", "auth", "status"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            # If gh CLI is authenticated, we can use it
            pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError(
                "GitHub authentication required. Either:\n"
                "1. Set GITHUB_TOKEN environment variable\n"
                "2. Configure github_token in .ci_fixer_bot.yml\n"
                "3. Authenticate with 'gh auth login'"
            )
    
    # Validate LLM provider settings
    if config.llm.provider == "openai" and not config.llm.api_key:
        raise ValueError("OpenAI API key required when using OpenAI provider. Set OPENAI_API_KEY or configure in yaml.")
    
    if config.llm.provider == "custom-endpoint" and not config.llm.endpoint:
        raise ValueError("Custom endpoint URL required when using custom-endpoint provider.")
    
    if config.llm.provider == "custom-cli" and not config.llm.command:
        raise ValueError("Custom command required when using custom-cli provider.")


def get_default_config() -> Config:
    """Get default configuration for documentation/examples."""
    return Config()