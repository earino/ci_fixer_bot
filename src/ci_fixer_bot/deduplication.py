"""
Issue deduplication module for ci_fixer_bot.

Prevents creating duplicate GitHub issues by comparing issue signatures
and searching for similar existing issues.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .github_client import GitHubClient
from .models import Issue


@dataclass
class IssueSignature:
    """Represents a unique signature for an issue based on its characteristics."""
    failure_type_hash: str
    error_pattern_hash: str
    affected_files_hash: str
    combined_hash: str
    
    @classmethod
    def create(cls, issue: Issue, failure_context: dict) -> 'IssueSignature':
        """Create an issue signature from an issue and its context."""
        
        # Extract key components for hashing
        failure_types = failure_context.get('failure_types', [])
        error_patterns = failure_context.get('error_patterns', [])
        affected_files = failure_context.get('affected_files', [])
        
        # Create normalized hashes
        failure_type_hash = cls._hash_list(failure_types)
        error_pattern_hash = cls._hash_error_patterns(error_patterns)
        affected_files_hash = cls._hash_file_paths(affected_files)
        
        # Create combined hash
        combined_input = f"{failure_type_hash}|{error_pattern_hash}|{affected_files_hash}"
        combined_hash = hashlib.sha256(combined_input.encode()).hexdigest()[:16]
        
        return cls(
            failure_type_hash=failure_type_hash,
            error_pattern_hash=error_pattern_hash,
            affected_files_hash=affected_files_hash,
            combined_hash=combined_hash
        )
    
    @staticmethod
    def _hash_list(items: List[str]) -> str:
        """Create a hash from a list of strings."""
        if not items:
            return "empty"
        
        # Sort and join to ensure consistent hashing
        normalized = "|".join(sorted(set(str(item).lower().strip() for item in items)))
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    @staticmethod
    def _hash_error_patterns(patterns: List[str]) -> str:
        """Create a hash from error patterns, focusing on the core error."""
        if not patterns:
            return "empty"
        
        # Normalize error patterns by removing specific details
        normalized_patterns = []
        for pattern in patterns:
            # Remove line numbers, specific file paths, timestamps
            normalized = re.sub(r':\d+', '', str(pattern))
            normalized = re.sub(r'/[^/\s]+/', '/*/', normalized)
            normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', normalized)
            normalized = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', normalized)
            normalized_patterns.append(normalized.lower().strip())
        
        return IssueSignature._hash_list(normalized_patterns)
    
    @staticmethod
    def _hash_file_paths(files: List[str]) -> str:
        """Create a hash from file paths, focusing on file names and extensions."""
        if not files:
            return "empty"
        
        # Extract just filenames and extensions for broader matching
        normalized_files = []
        for file_path in files:
            # Get just the filename and extension
            filename = file_path.split('/')[-1] if '/' in file_path else file_path
            normalized_files.append(filename.lower())
        
        return IssueSignature._hash_list(normalized_files)


@dataclass
class DuplicationResult:
    """Result of a duplication check."""
    is_duplicate: bool
    existing_issue: Optional[Dict] = None
    similarity_score: float = 0.0
    reason: str = ""


class IssueDeduplicator:
    """Handles detection and prevention of duplicate issues."""
    
    def __init__(
        self,
        github_client: GitHubClient,
        similarity_threshold: float = 0.8,
        exact_match_threshold: float = 0.95
    ):
        """
        Initialize the deduplicator.
        
        Args:
            github_client: GitHub API client
            similarity_threshold: Minimum similarity score to consider a duplicate (0.0-1.0)
            exact_match_threshold: Threshold for exact matches (0.0-1.0)
        """
        self.github_client = github_client
        self.similarity_threshold = similarity_threshold
        self.exact_match_threshold = exact_match_threshold
    
    def check_for_duplicate(
        self,
        owner: str,
        repo: str,
        new_issue: Issue,
        failure_context: dict
    ) -> DuplicationResult:
        """
        Check if an issue is a duplicate of existing issues.
        
        Args:
            owner: Repository owner
            repo: Repository name
            new_issue: Issue to check for duplication
            failure_context: Context about the failures that generated this issue
            
        Returns:
            DuplicationResult indicating if this is a duplicate
        """
        
        # Create signature for the new issue
        new_signature = IssueSignature.create(new_issue, failure_context)
        
        # Search for potentially similar issues
        candidate_issues = self._find_candidate_issues(owner, repo, new_issue, failure_context)
        
        if not candidate_issues:
            return DuplicationResult(
                is_duplicate=False,
                reason="No similar issues found"
            )
        
        # Calculate similarity with each candidate
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_issues:
            score = self._calculate_similarity(new_issue, candidate, new_signature, failure_context)
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Determine if this is a duplicate
        is_duplicate = best_score >= self.similarity_threshold
        
        if is_duplicate:
            reason = f"Similar issue found (similarity: {best_score:.2f})"
            if best_score >= self.exact_match_threshold:
                reason = f"Exact match found (similarity: {best_score:.2f})"
        else:
            reason = f"No duplicates found (best match: {best_score:.2f})"
        
        return DuplicationResult(
            is_duplicate=is_duplicate,
            existing_issue=best_match if is_duplicate else None,
            similarity_score=best_score,
            reason=reason
        )
    
    def _find_candidate_issues(
        self,
        owner: str,
        repo: str,
        new_issue: Issue,
        failure_context: dict
    ) -> List[Dict]:
        """Find issues that might be duplicates using GitHub search."""
        
        candidates = []
        
        # Search strategy 1: Look for issues with ci-fix label and similar risk level
        search_queries = [
            f"label:ci-fix label:{new_issue.risk_level}-fix",
            "label:ci-fix",
            "ci failure"
        ]
        
        # Add specific search terms based on failure context
        failure_types = failure_context.get('failure_types', [])
        if failure_types:
            for failure_type in failure_types[:2]:  # Limit to avoid too many queries
                search_queries.append(f"label:type-{failure_type}")
        
        # Add error pattern based searches
        error_patterns = failure_context.get('error_patterns', [])
        if error_patterns:
            # Use first error pattern for search (cleaned up)
            first_pattern = error_patterns[0]
            cleaned_pattern = self._clean_search_term(first_pattern)
            if len(cleaned_pattern) > 3:  # Only search if meaningful
                search_queries.append(f'"{cleaned_pattern}"')
        
        # Execute searches and collect unique candidates
        seen_numbers = set()
        
        for query in search_queries:
            try:
                issues = self.github_client.search_issues(
                    owner=owner,
                    repo=repo,
                    query=query,
                    state="open"
                )
                
                for issue in issues:
                    if issue['number'] not in seen_numbers:
                        seen_numbers.add(issue['number'])
                        candidates.append(issue)
                        
                        # Limit candidates to prevent too much processing
                        if len(candidates) >= 20:
                            break
                            
            except Exception:
                # Continue with other searches if one fails
                continue
        
        return candidates
    
    def _clean_search_term(self, term: str) -> str:
        """Clean a search term for GitHub search."""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s-]', ' ', str(term))
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove very short words and numbers
        words = [w for w in cleaned.split() if len(w) > 2 and not w.isdigit()]
        
        return ' '.join(words[:5])  # Limit to first 5 meaningful words
    
    def _calculate_similarity(
        self,
        new_issue: Issue,
        existing_issue: Dict,
        new_signature: IssueSignature,
        failure_context: dict
    ) -> float:
        """Calculate similarity score between new and existing issue."""
        
        scores = []
        
        # 1. Title similarity (30% weight)
        title_similarity = self._text_similarity(
            new_issue.title.lower(),
            existing_issue.get('title', '').lower()
        )
        scores.append(('title', title_similarity, 0.3))
        
        # 2. Label similarity (25% weight)
        existing_labels = [label['name'] for label in existing_issue.get('labels', [])]
        label_similarity = self._list_similarity(new_issue.labels, existing_labels)
        scores.append(('labels', label_similarity, 0.25))
        
        # 3. Risk level similarity (20% weight)
        risk_similarity = 1.0 if new_issue.risk_level in str(existing_issue.get('title', '')).lower() else 0.0
        scores.append(('risk', risk_similarity, 0.2))
        
        # 4. Body content similarity (15% weight)
        body_similarity = self._text_similarity(
            new_issue.body.lower(),
            existing_issue.get('body', '').lower()
        )
        scores.append(('body', body_similarity, 0.15))
        
        # 5. Failure type similarity (10% weight)
        failure_types = failure_context.get('failure_types', [])
        existing_body = existing_issue.get('body', '').lower()
        
        failure_type_similarity = 0.0
        if failure_types:
            for failure_type in failure_types:
                if failure_type in existing_body:
                    failure_type_similarity = 1.0
                    break
        
        scores.append(('failure_type', failure_type_similarity, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Remove very common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists."""
        if not list1 or not list2:
            return 0.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0