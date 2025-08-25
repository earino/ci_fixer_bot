# Contributing to ci_fixer_bot

Thank you for your interest in contributing to ci_fixer_bot! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.9+ 
- Git
- GitHub CLI (`gh`) - optional but recommended

### Local Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/earino/ci_fixer_bot.git
   cd ci_fixer_bot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,test]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Development Workflow

### Code Quality Standards

We maintain high code quality through automated tools:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security analysis

### Running Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/ci_fixer_bot/

# Security scan
bandit -r src/
```

### Testing

We use pytest for testing with comprehensive coverage requirements:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ci_fixer_bot --cov-report=html

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run pre-commit checks**
   ```bash
   pre-commit run --all-files
   ```

4. **Run tests**
   ```bash
   pytest
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**
   ```bash
   git push -u origin feature/your-feature-name
   gh pr create --title "Your PR title" --body "Description of changes"
   ```

## Commit Message Guidelines

We follow conventional commit format:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## Code Architecture

### Key Components

- **`core.py`** - Main orchestration logic
- **`analyzers.py`** - CI failure analysis engine
- **`deduplication.py`** - Issue deduplication system
- **`github_client.py`** - GitHub API interactions
- **`llm_providers.py`** - LLM provider abstractions
- **`risk_assessor.py`** - Risk classification system
- **`config.py`** - Configuration management

### Design Principles

1. **Risk-First Design** - Every component considers safety implications
2. **LLM-Agnostic** - Support multiple LLM providers
3. **Senior-Engineer-First** - Optimize for experienced developers
4. **Boring is Beautiful** - Use proven, reliable patterns

## Adding New Features

### Before Starting

1. Check existing issues and discussions
2. Consider creating an issue to discuss the feature
3. Ensure the feature aligns with our design principles

### Implementation Guidelines

1. **Maintain backward compatibility**
2. **Add comprehensive tests**
3. **Update configuration if needed**
4. **Document new functionality**
5. **Consider security implications**

### Specific Areas

#### Adding New LLM Providers

1. Implement the `LLMProvider` interface
2. Add configuration options
3. Add comprehensive error handling
4. Test with various prompts and scenarios

#### Extending Risk Assessment

1. Update `risk_assessor.py` patterns
2. Add configuration for new risk rules
3. Ensure patterns are well-tested
4. Document reasoning for risk classifications

#### Improving Deduplication

1. Consider similarity algorithm improvements
2. Add new comparison factors
3. Maintain configurable thresholds
4. Test with real-world issue data

## Security Considerations

- **Never log secrets or API keys**
- **Validate all external inputs**
- **Use secure defaults in configurations**
- **Follow principle of least privilege**
- **Consider rate limiting implications**

## Documentation

- Update README.md for user-facing changes
- Add docstrings for all public functions
- Update configuration examples
- Consider adding usage examples

## Getting Help

- Create issues for bugs or feature requests
- Start discussions for architectural questions
- Check existing issues before creating new ones
- Tag maintainers for urgent issues

## Release Process

Releases are automated but follow this process:

1. **Version Bump** - Update version in `pyproject.toml`
2. **Changelog** - Auto-generated from conventional commits
3. **Testing** - Full CI/CD pipeline runs
4. **Release** - GitHub release with assets
5. **PyPI** - Automatic publication for stable releases
6. **Docker** - Container images published to GHCR

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professionalism

Thank you for contributing to ci_fixer_bot! 🚀