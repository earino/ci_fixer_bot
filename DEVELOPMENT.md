# Development Guide

This guide covers the technical aspects of developing and maintaining ci_fixer_bot.

## Architecture Overview

```
ci_fixer_bot/
├── src/ci_fixer_bot/
│   ├── core.py              # Main orchestration
│   ├── analyzers.py         # Failure analysis engine
│   ├── deduplication.py     # Issue deduplication
│   ├── github_client.py     # GitHub API client
│   ├── llm_providers.py     # LLM abstractions
│   ├── risk_assessor.py     # Risk classification
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models
│   └── cli.py              # Command-line interface
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
└── .github/workflows/      # CI/CD pipelines
```

## CI/CD Infrastructure

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Multi-Python version testing (3.9-3.12)
   - Code quality checks (black, isort, flake8, mypy)
   - Security scanning (bandit, safety)
   - Coverage reporting
   - Integration tests

2. **Security Scanning** (`.github/workflows/security.yml`)
   - Weekly scheduled security scans
   - CodeQL analysis for vulnerability detection
   - Dependency vulnerability scanning
   - Security report generation

3. **Release Automation** (`.github/workflows/release.yml`)
   - Automated version management
   - Package building and testing
   - GitHub release creation
   - PyPI publishing
   - Docker image building

4. **Dependency Management** (`.github/workflows/dependabot-auto-merge.yml`)
   - Automatic dependency updates
   - Auto-merge for minor/patch updates
   - Manual review for major updates

### Pre-commit Configuration

Local development uses pre-commit hooks to ensure code quality:

- Trailing whitespace removal
- End-of-file fixing
- JSON/YAML validation
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit)

## Local Development

### Setup Commands

```bash
# Initial setup
git clone https://github.com/earino/ci_fixer_bot.git
cd ci_fixer_bot
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test]"
pre-commit install

# Development workflow
git checkout -b feature/my-feature
# Make changes...
pre-commit run --all-files  # Run all checks
pytest                       # Run tests
git commit -m "feat: add new feature"
git push -u origin feature/my-feature
```

### Testing Strategy

#### Unit Tests
- Located in `tests/unit/`
- Test individual components in isolation
- Mock external dependencies
- Fast execution, comprehensive coverage

#### Integration Tests  
- Located in `tests/integration/`
- Test component interactions
- Use real APIs with test data
- Verify end-to-end workflows

#### Test Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=src/ci_fixer_bot --cov-report=html --cov-report=term-missing"
```

### Code Quality Tools

#### Formatting
```bash
black src/ tests/           # Format code
isort src/ tests/          # Sort imports
```

#### Linting
```bash
flake8 src/ tests/         # Style linting
mypy src/ci_fixer_bot/     # Type checking
bandit -r src/             # Security analysis
```

#### Configuration Files
- `.flake8` - Linting rules and exclusions
- `pyproject.toml` - Tool configurations
- `.pre-commit-config.yaml` - Pre-commit hook setup

## Deployment

### Docker Containerization

```dockerfile
# Dockerfile provides containerized deployment
FROM python:3.11-slim
# ... (see Dockerfile for full configuration)
```

```bash
# Build and run container
docker build -t ci_fixer_bot .
docker run --rm ci_fixer_bot --help

# Using published images
docker pull ghcr.io/earino/ci_fixer_bot:latest
```

### Release Process

1. **Manual Release**
   ```bash
   # Trigger via GitHub Actions
   gh workflow run release.yml -f version=0.2.0
   ```

2. **Tag-based Release**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   # Automatically triggers release workflow
   ```

3. **Release Artifacts**
   - GitHub release with changelog
   - PyPI package (for stable releases)
   - Docker images on GitHub Container Registry
   - Distribution archives

## Security Considerations

### Secrets Management
- Never commit API keys or tokens
- Use GitHub Secrets for CI/CD
- Environment variable configuration
- Secure defaults in all components

### Vulnerability Scanning
- Bandit for Python security issues
- Safety for known vulnerable dependencies  
- CodeQL for advanced static analysis
- Regular dependency updates via Dependabot

### API Security
- Rate limiting awareness
- Input validation and sanitization
- Secure GitHub API interactions
- Principle of least privilege

## Configuration Management

### Environment Variables
```bash
export GITHUB_TOKEN="ghp_xxxx"
export OPENAI_API_KEY="sk-xxxx"  
export SLACK_WEBHOOK_URL="https://hooks.slack.com/xxxx"
```

### Configuration Files
```yaml
# .ci_fixer_bot.yml
analyze_runs: 10
focus: "tactical"
min_priority: 50

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1

deduplication:
  enabled: true
  similarity_threshold: 0.8
  update_existing: true

notifications:
  slack_webhook: "${SLACK_WEBHOOK_URL}"
  enabled: true
```

## Monitoring and Observability

### GitHub Actions Monitoring
- Workflow status badges in README
- Failed build notifications
- Security scan alerts
- Dependency update notifications

### Application Monitoring
- Structured logging with risk context
- GitHub API rate limit tracking
- LLM provider usage metrics
- Issue creation success rates

## Performance Considerations

### GitHub API Optimization
- Request batching where possible
- Response caching for repeated calls
- Rate limit awareness and backoff
- Efficient search query construction

### LLM Provider Optimization
- Context window management
- Response parsing optimization
- Provider failover strategies
- Token usage monitoring

## Troubleshooting

### Common Issues

1. **GitHub API Rate Limits**
   - Check remaining quota: `gh api rate_limit`
   - Use authenticated requests
   - Implement exponential backoff

2. **LLM Provider Failures**
   - Configure fallback providers
   - Validate API keys and endpoints
   - Check service status pages

3. **Pre-commit Failures**
   - Run `pre-commit run --all-files`
   - Check individual tool configurations
   - Update hook versions if needed

### Debug Commands

```bash
# Check configuration
python -c "from ci_fixer_bot.config import load_config; print(load_config())"

# Test GitHub connectivity
python -c "from ci_fixer_bot.github_client import GitHubClient; client = GitHubClient(); print(client.get_repository_info('earino', 'ci_fixer_bot'))"

# Validate environment
python -m ci_fixer_bot --help
```

This development infrastructure ensures code quality, security, and reliable deployment of ci_fixer_bot. 🛠️