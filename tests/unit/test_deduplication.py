"""
Unit tests for deduplication module.
"""

import pytest

from ci_fixer_bot.deduplication import IssueDeduplicator, IssueSignature
from ci_fixer_bot.models import Issue


class MockGitHubClient:
    """Mock GitHub client for testing."""
    
    def __init__(self, mock_issues=None):
        self.mock_issues = mock_issues or []
        
    def search_issues(self, owner, repo, query, state="open"):
        return self.mock_issues


class TestIssueSignature:
    """Test issue signature creation and hashing."""
    
    def test_signature_creation(self):
        """Test that signatures are created correctly."""
        issue = Issue(
            title="Fix test failure",
            body="Test body",
            risk_level="tactical",
            priority=60,
            estimated_effort="2 hours",
            required_expertise="mid"
        )
        
        context = {
            'failure_types': ['test_failure'],
            'error_patterns': ['AssertionError'],
            'affected_files': ['test_auth.py'],
            'failure_count': 1
        }
        
        signature = IssueSignature.create(issue, context)
        
        assert signature.combined_hash is not None
        assert len(signature.combined_hash) == 16
        assert signature.failure_type_hash is not None
        assert signature.error_pattern_hash is not None
        assert signature.affected_files_hash is not None
        
    def test_signature_consistency(self):
        """Test that identical issues produce identical signatures."""
        issue = Issue(
            title="Fix test failure",
            body="Test body", 
            risk_level="tactical",
            priority=60,
            estimated_effort="2 hours",
            required_expertise="mid"
        )
        
        context = {
            'failure_types': ['test_failure'],
            'error_patterns': ['AssertionError'],
            'affected_files': ['test_auth.py'],
        }
        
        sig1 = IssueSignature.create(issue, context)
        sig2 = IssueSignature.create(issue, context)
        
        assert sig1.combined_hash == sig2.combined_hash
        
    def test_signature_differentiation(self):
        """Test that different issues produce different signatures."""
        issue = Issue(
            title="Fix test failure",
            body="Test body",
            risk_level="tactical", 
            priority=60,
            estimated_effort="2 hours",
            required_expertise="mid"
        )
        
        context1 = {
            'failure_types': ['test_failure'],
            'error_patterns': ['AssertionError'],
            'affected_files': ['test_auth.py'],
        }
        
        context2 = {
            'failure_types': ['linting'],
            'error_patterns': ['trailing whitespace'],
            'affected_files': ['main.py'],
        }
        
        sig1 = IssueSignature.create(issue, context1)
        sig2 = IssueSignature.create(issue, context2)
        
        assert sig1.combined_hash != sig2.combined_hash


class TestIssueDeduplicator:
    """Test issue deduplication functionality."""
    
    def test_text_similarity_identical(self):
        """Test text similarity with identical strings."""
        deduplicator = IssueDeduplicator(MockGitHubClient())
        
        similarity = deduplicator._text_similarity(
            "fix auth test failure",
            "fix auth test failure"
        )
        
        assert similarity == 1.0
        
    def test_text_similarity_different(self):
        """Test text similarity with different strings."""
        deduplicator = IssueDeduplicator(MockGitHubClient())
        
        similarity = deduplicator._text_similarity(
            "fix authentication",
            "linting whitespace error"
        )
        
        assert similarity < 0.3
        
    def test_text_similarity_similar(self):
        """Test text similarity with similar strings."""
        deduplicator = IssueDeduplicator(MockGitHubClient())
        
        similarity = deduplicator._text_similarity(
            "fix authentication test failure",
            "fix auth test failure"
        )
        
        assert similarity > 0.5
        
    def test_no_duplicates_found(self):
        """Test when no duplicates are found."""
        github_client = MockGitHubClient(mock_issues=[])
        deduplicator = IssueDeduplicator(github_client)
        
        issue = Issue(
            title="Fix new test failure",
            body="New test body",
            risk_level="tactical",
            priority=60,
            estimated_effort="2 hours", 
            required_expertise="mid"
        )
        
        context = {'failure_types': ['test_failure']}
        
        result = deduplicator.check_for_duplicate("owner", "repo", issue, context)
        
        assert not result.is_duplicate
        assert result.existing_issue is None
        assert result.similarity_score == 0.0
        
    def test_duplicate_found(self):
        """Test when a duplicate is found."""
        mock_existing_issue = {
            'number': 123,
            'title': '🟡 [TACTICAL] Fix test failure in auth module',
            'body': 'Test failures in authentication',
            'labels': [
                {'name': 'ci-fix'},
                {'name': 'tactical-fix'},
                {'name': 'type-test_failure'}
            ]
        }
        
        github_client = MockGitHubClient(mock_issues=[mock_existing_issue])
        deduplicator = IssueDeduplicator(github_client, similarity_threshold=0.5)
        
        issue = Issue(
            title="🟡 [TACTICAL] Fix authentication test failures", 
            body="Fix test failures in auth module",
            risk_level="tactical",
            priority=60,
            estimated_effort="2 hours",
            required_expertise="mid",
            labels=['ci-fix', 'tactical-fix', 'type-test_failure']
        )
        
        context = {
            'failure_types': ['test_failure'],
            'error_patterns': ['AssertionError'],
            'affected_files': ['test_auth.py']
        }
        
        result = deduplicator.check_for_duplicate("owner", "repo", issue, context)
        
        assert result.is_duplicate
        assert result.existing_issue == mock_existing_issue
        assert result.similarity_score >= 0.5