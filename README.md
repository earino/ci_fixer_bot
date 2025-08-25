# 🤖 ci_fixer_bot

**Intelligent CI failure analysis tool that creates risk-aware, actionable GitHub issues**

Transform CI noise into prioritized, risk-assessed action items that can be safely delegated or strategically planned.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

## 🎯 What ci_fixer_bot Does

Instead of just telling you "CI is broken", ci_fixer_bot:

- **🟢 Identifies SAFE fixes** that any developer can handle (linting, formatting)
- **🟡 Flags TACTICAL issues** that need testing but are low-risk  
- **🔴 Highlights STRATEGIC problems** that require senior engineering planning
- **🧠 Uses AI analysis** to understand root causes, not just symptoms
- **📋 Creates actionable issues** with clear next steps and risk assessment

## ✨ Key Features

- **Risk-Aware Analysis**: Every issue labeled with safety level and required expertise
- **LLM Provider Flexibility**: Use Claude, OpenAI, local LLMs, or custom endpoints
- **Pattern Recognition**: Identifies flaky tests, environment mismatches, recurring failures
- **Smart Delegation**: Clearly marks what junior devs can safely fix
- **Knowledge Building**: Learns from successful fixes to improve future analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/earino/ci_fixer_bot.git
cd ci_fixer_bot

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Analyze a repository and create GitHub issues
ci_fixer_bot https://github.com/owner/repo

# Preview issues without creating them
ci_fixer_bot https://github.com/owner/repo --dry-run

# Focus on safe fixes only
ci_fixer_bot https://github.com/owner/repo --focus=safe

# Use different LLM provider
ci_fixer_bot https://github.com/owner/repo --llm-provider=openai
```

## 🔧 Configuration

Create `.ci_fixer_bot.yml` in your project or home directory:

```yaml
# LLM Provider (choose one)
llm:
  # Option 1: Claude CLI (default)
  provider: claude-cli
  
  # Option 2: OpenAI API
  # provider: openai
  # api_key: ${OPENAI_API_KEY}
  # model: gpt-4
  
  # Option 3: Local LLM (Ollama)
  # provider: ollama
  # model: codellama
  
  # Option 4: Custom endpoint
  # provider: custom-endpoint
  # endpoint: http://localhost:1234/v1/completions

# Risk assessment overrides
risk_overrides:
  "payment_*.py": "strategic"    # Payment code always high-risk
  "test_*.py": "safe"           # Test fixes usually safe
  "*_migration.rb": "strategic"  # Database migrations need planning

# Team assignments
team_mapping:
  frontend: ["*.jsx", "*.tsx", "*.css"]
  backend: ["*.py", "*.rb"]
  devops: [".github/workflows/*", "Dockerfile"]
```

## 🎭 Example Output

### 🟢 Safe Fix Issue
```markdown
# 🟢 [SAFE] Fix 127 ESLint style violations

**Risk Level**: None - Only affects development
**Effort**: 15 minutes  
**Can be fixed by**: Anyone

## Quick Fix
```bash
npm run lint:fix
git commit -am "fix: Auto-fix ESLint violations"
```

Perfect for new team members or during cooldown time.
```

### 🔴 Strategic Issue
```markdown
# 🔴 [STRATEGIC] Database version mismatch across environments

**Risk Level**: HIGH - Could affect production
**Effort**: 4-6 hours + coordination
**Required**: Senior Engineer + DevOps

## ⚠️ DO NOT FIX WITHOUT PLANNING

CI uses PostgreSQL 15, production uses PostgreSQL 14.
This needs careful coordination and testing.

## Required Before Fixing
- [ ] Audit production compatibility
- [ ] Plan maintenance window
- [ ] Test rollback procedures
```

## 📊 Supported CI Platforms

- ✅ **GitHub Actions** (primary)
- 🚧 CircleCI (planned)
- 🚧 GitLab CI (planned) 
- 🚧 Travis CI (planned)

## 🧠 LLM Provider Support

- ✅ **Claude CLI** (default)
- ✅ **OpenAI API** (GPT-3.5, GPT-4)
- ✅ **Ollama** (local models)
- ✅ **LM Studio** (local GUI)
- ✅ **Custom endpoints** (Azure OpenAI, etc.)
- ✅ **No LLM mode** (pattern matching only)

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📚 Documentation

- [ROADMAP.md](ROADMAP.md) - Comprehensive product specification
- [examples/](examples/) - Usage examples and advanced configuration
- [API Documentation](docs/api.md) - For integrating with ci_fixer_bot

## 🤝 Contributing

This is proprietary software. See [LICENSE](LICENSE) for terms.

For feature requests or bug reports, please open a GitHub issue.

## 📈 Roadmap

- **Week 1**: MVP with GitHub Actions support
- **Week 2**: Pattern detection and knowledge base
- **Week 3**: Auto-fix capabilities for safe issues
- **Month 2**: Multi-CI platform support and analytics

See [ROADMAP.md](ROADMAP.md) for detailed development plan.

## 📞 Support

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/earino/ci_fixer_bot/issues)
- 💡 **Feature requests**: [GitHub Discussions](https://github.com/earino/ci_fixer_bot/discussions)
- 📧 **Commercial inquiries**: earino@gmail.com

---

**Making CI failures less painful, one intelligent issue at a time.** 🚀

*Built with ❤️ by senior engineers who've spent too much time debugging CI*