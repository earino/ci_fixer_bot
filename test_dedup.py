#!/usr/bin/env python
"""Test script to debug deduplication."""

import logging
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.ci_fixer_bot.config import load_config
from src.ci_fixer_bot.github_client import GitHubClient
from src.ci_fixer_bot.embedding_deduplicator import create_embedding_deduplicator
from src.ci_fixer_bot.models import Issue

# Load config
config = load_config(Path(".ci_fixer_bot_test.yml"))

# Create GitHub client
github_client = GitHubClient(token=config.github_token)

# Create deduplicator
print("Creating deduplicator...")
deduplicator = create_embedding_deduplicator(config, github_client)

# Index existing issues
print("\nIndexing existing issues...")
count = deduplicator.index_existing_issues("earino", "ci_fixer_bot", labels=["ci-fix"])
print(f"Indexed {count} issues")

# Create a test issue to check for duplicates
test_issue = Issue(
    title="🟡 [TACTICAL] Fix CI failure",
    body="This is a test issue to check deduplication",
    risk_level="tactical",
    priority=50,
    estimated_effort="1-3 hours",
    required_expertise="mid",
    labels=["ci-fix", "type-test"],
)

print(f"\nChecking for duplicates of: {test_issue.title}")
print(f"Deduplicator threshold: {deduplicator.threshold}")
print(f"Vector store size: {deduplicator.vector_store.size()}")

# Try searching directly without threshold
test_text = deduplicator._extract_issue_text({
    "title": test_issue.title,
    "body": test_issue.body,
    "labels": test_issue.labels
})
test_embedding = deduplicator.provider.embed_text(test_text)

# Search without threshold to see all scores
all_results = deduplicator.vector_store.search(test_embedding, k=10, threshold=0.0)
print(f"\nAll similarity scores (no threshold):")
for score, metadata in all_results[:10]:
    print(f"  #{metadata['number']}: {metadata['title'][:50]}... - Score: {score:.3f}")

print(f"\nNow checking with threshold {deduplicator.threshold}:")
result = deduplicator.check_duplicate(test_issue)

print(f"\nIs duplicate: {result.is_duplicate}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Number of similar issues found: {len(result.similar_issues)}")

if result.similar_issues:
    print("\nSimilar issues found:")
    for issue in result.similar_issues[:5]:  # Show top 5
        print(f"  #{issue.issue_number}: {issue.title}")
        print(f"    Similarity: {issue.similarity_score:.2%}")

if result.best_match:
    print(f"\nBest match: #{result.best_match.issue_number} - {result.best_match.title}")
    print(f"Similarity: {result.best_match.similarity_score:.2%}")