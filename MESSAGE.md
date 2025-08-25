
⏺ 📨 Message to Future AI Developer

  Context: You're Inheriting ci_fixer_bot

  Hey future AI! You're now the maintainer of ci_fixer_bot - an intelligent CI failure analysis tool that
   creates risk-aware GitHub issues. Here's what past-me learned building this:

  ---
  🎯 The Core Mission

  This isn't just another CI tool. It's built for senior engineers who need to:
  - Delegate safely - Mark what juniors can fix without risk
  - Prioritize strategically - Focus on what actually matters
  - Avoid disasters - Never suggest dangerous fixes

  🧠 The Key Insight

  The magic is in the risk assessment. Everything flows from classifying fixes as:
  - 🟢 SAFE - Linting, docs, formatting (delegate to anyone)
  - 🟡 TACTICAL - Tests, minor code (needs verification)
  - 🔴 STRATEGIC - DB, auth, payments (requires planning)

  🏗️ Architecture Decisions Made

  LLM Provider Abstraction - This was crucial. Users want:
  - Claude CLI (default)
  - OpenAI API (powerful)
  - Ollama (private/local)
  - Custom endpoints (enterprise)
  - No LLM (pattern-matching fallback)

  Risk-First Design - Every component asks "is this safe to fix?" before anything else.

  GitHub-Native - Creates actual issues, not reports. Issues are where work gets done.

  ⚠️ Critical Implementation Notes

  1. Risk Patterns (risk_assessor.py) - These are GOLD. Spent tons of time getting the pattern matching
  right. Don't break them casually.
  2. LLM Prompts (analyzers.py) - The JSON-structured prompts work well. Keep the "senior engineer
  analyzing CI failures" persona.
  3. Fallback Logic - Always have pattern-matching fallbacks when LLMs fail. Never crash, always provide
  value.
  4. GitHub API Rate Limits - Built-in retry logic, but watch for rate limits with large repos.

  🚀 What's Working Well

  - CLI UX - Rich formatting, clear output, good error messages
  - Provider Flexibility - Easy to swap LLMs without breaking anything
  - Issue Templates - Generate issues that senior engineers actually want to read
  - Configuration System - YAML config with env var overrides

  🔧 What Needs Work (Priority Order)

  1. Pattern Detection - The failure grouping logic is basic. Could be much smarter.
  2. Knowledge Base - Stub implementation exists, needs real persistence
  3. Auto-Fix PRs - Framework is there, need to implement safe fixes
  4. Multi-CI Support - Only GitHub Actions works, need CircleCI/GitLab
  5. Analytics - Tracking fix success rates, time savings, etc.

  🎨 Design Philosophy

  Boring is Beautiful - Used Python stdlib wherever possible. No fancy frameworks.

  Senior-Engineer-First - Every decision was "would a 10-year senior engineer find this useful?"

  Risk-Obsessed - When in doubt, mark it strategic. False negatives (missing easy fixes) are way better
  than false positives (suggesting dangerous fixes).

  LLM-Agnostic - Don't assume everyone has Claude/OpenAI access.

  ⚡ Quick Wins for You

  1. Test Suite - There's a basic test structure but needs real tests
  2. Error Patterns - Add more patterns to risk_assessor.py as you see real failures
  3. Issue Deduplication - Framework exists but logic is basic
  4. Config Validation - Add better validation with helpful error messages

  🐛 Known Issues

  - Log parsing could be more robust (some CI platforms have weird formats)
  - GitHub API token detection sometimes flaky
  - Large repos might hit rate limits
  - Some LLM providers have different JSON response formats

  📚 Key Files to Understand

  1. analyzers.py - The brain. Parses failures → LLM → issues
  2. risk_assessor.py - The soul. Safe/tactical/strategic classification
  3. llm_providers.py - The flexibility. Multiple LLM support
  4. cli.py - The face. What users actually interact with

  🎯 Success Metrics

  Users should say:
  - "Finally, CI issues I can actually delegate!"
  - "It caught that dangerous DB change before I merged it"
  - "My junior devs can fix the safe stuff without bothering me"

  💭 Final Thoughts

  This codebase is functional but not finished. The core architecture is solid, but there's tons of room
  for intelligence improvements.

  The risk assessment patterns are the most valuable part - they encode real senior engineering knowledge
   about what's safe to change.

  Stay true to the mission: make CI failures less painful by making them more actionable. 🚀

  ---
  Good luck! You've got a solid foundation to build on.

  - Past AI (ci_fixer_bot architect) 🏗️
