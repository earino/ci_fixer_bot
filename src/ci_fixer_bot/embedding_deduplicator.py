"""
Embeddings-based issue deduplication using semantic similarity.

This module replaces the brittle text-matching deduplication with
intelligent semantic matching using embedding vectors.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .embedding_providers import EmbeddingProvider, create_embedding_provider
from .github_client import GitHubClient
from .models import Issue, DuplicationResult
from .vector_store import VectorStore, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class SimilarIssue:
    """Represents a similar issue found during deduplication."""
    issue_number: int
    title: str
    url: str
    similarity_score: float
    state: str
    created_at: str
    updated_at: str
    labels: List[str]
    
    @property
    def days_old(self) -> int:
        """Calculate how many days old this issue is."""
        created = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        return (datetime.now(created.tzinfo) - created).days


class EmbeddingDeduplicator:
    """
    Issue deduplicator using embedding-based semantic similarity.
    
    This replaces the old regex-based deduplication with intelligent
    semantic matching that understands when issues are describing
    the same problem even with different wording.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        github_client: GitHubClient,
        similarity_threshold: float = 0.85,
        vector_store_type: str = "memory",
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize the embedding-based deduplicator.
        
        Args:
            embedding_provider: Provider for generating embeddings
            github_client: Client for GitHub API operations
            similarity_threshold: Minimum similarity to consider duplicate (0-1)
            vector_store_type: Type of vector store ("memory" or "faiss")
            cache_dir: Directory for caching embeddings
            cache_ttl_hours: Hours before cache expires
        """
        self.provider = embedding_provider
        self.github_client = github_client
        self.threshold = similarity_threshold
        self.cache_dir = cache_dir or Path(".ci_fixer_bot_cache")
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Create vector store
        self.vector_store = create_vector_store(
            store_type=vector_store_type,
            dimension=embedding_provider.get_embedding_dim()
        )
        
        # Track indexed issues
        self.indexed_issues: Dict[int, Dict[str, Any]] = {}
        self.last_index_time: Optional[datetime] = None
        
        logger.info(
            f"Initialized EmbeddingDeduplicator: "
            f"threshold={similarity_threshold}, store={vector_store_type}"
        )
    
    def _extract_issue_text(self, issue: Dict[str, Any]) -> str:
        """
        Extract and combine relevant text from an issue for embedding.
        
        Args:
            issue: GitHub issue data
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Title is most important
        title = issue.get("title", "")
        if title:
            parts.append(f"Title: {title}")
        
        # Body contains error details
        body = issue.get("body", "")
        if body:
            # Truncate very long bodies
            if len(body) > 2000:
                body = body[:2000] + "..."
            parts.append(f"Description: {body}")
        
        # Labels provide context
        labels = issue.get("labels", [])
        if labels:
            label_names = [
                label["name"] if isinstance(label, dict) else label 
                for label in labels
            ]
            parts.append(f"Labels: {', '.join(label_names)}")
        
        # Combine with newlines
        combined = "\n".join(parts)
        
        # Ensure we have something to embed
        return combined if combined else "Empty issue"
    
    def index_existing_issues(
        self, 
        owner: str, 
        repo: str,
        labels: Optional[List[str]] = None,
        state: str = "open"
    ) -> int:
        """
        Index existing issues from a repository into the vector store.
        
        Args:
            owner: Repository owner
            repo: Repository name
            labels: Optional labels to filter issues
            state: Issue state ("open", "closed", or "all")
            
        Returns:
            Number of issues indexed
        """
        logger.info(f"Indexing {state} issues from {owner}/{repo}")
        
        # Check cache first
        cache_file = self._get_cache_file(owner, repo)
        if self._load_cache(cache_file):
            logger.info(f"Loaded {len(self.indexed_issues)} issues from cache")
            return len(self.indexed_issues)
        
        # Fetch issues from GitHub
        issues = self.github_client.list_issues(
            owner=owner,
            repo=repo,
            state=state,
            labels=labels
        )
        
        if not issues:
            logger.info("No issues to index")
            return 0
        
        # Extract text for each issue
        texts = []
        metadata = []
        
        for issue in issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue
            
            text = self._extract_issue_text(issue)
            texts.append(text)
            
            # Store metadata
            meta = {
                "number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "labels": [
                    label["name"] if isinstance(label, dict) else label
                    for label in issue.get("labels", [])
                ]
            }
            metadata.append(meta)
            self.indexed_issues[issue["number"]] = meta
        
        if texts:
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(texts)} issues...")
            embeddings = self.provider.embed_batch(texts)
            
            # Add to vector store
            self.vector_store.add(embeddings, metadata)
            
            # Save cache
            self._save_cache(cache_file)
            
            logger.info(f"Indexed {len(texts)} issues")
        
        self.last_index_time = datetime.now()
        return len(texts)
    
    def check_duplicate(
        self,
        issue: Issue,
        context: Optional[Dict[str, Any]] = None
    ) -> DuplicationResult:
        """
        Check if an issue is a duplicate using semantic similarity.
        
        Args:
            issue: The issue to check
            context: Optional context about the issue
            
        Returns:
            DuplicationResult with duplicate status and similar issues
        """
        if self.vector_store.size() == 0:
            logger.warning("No issues indexed, skipping deduplication")
            return DuplicationResult(
                is_duplicate=False,
                similar_issues=[],
                best_match=None,
                confidence=0.0
            )
        
        # Prepare issue text for embedding
        issue_dict = {
            "title": issue.title,
            "body": issue.body,  # Use body, not description
            "labels": getattr(issue, "labels", [])
        }
        text = self._extract_issue_text(issue_dict)
        
        # Generate embedding
        embedding = self.provider.embed_text(text)
        
        # Search for similar issues
        results = self.vector_store.search(
            query_embedding=embedding,
            k=10,  # Get top 10 candidates
            threshold=self.threshold
        )
        
        if not results:
            return DuplicationResult(
                is_duplicate=False,
                similar_issues=[],
                best_match=None,
                confidence=0.0
            )
        
        # Convert results to SimilarIssue objects
        similar_issues = []
        for score, metadata in results:
            similar = SimilarIssue(
                issue_number=metadata["number"],
                title=metadata["title"],
                url=metadata["url"],
                similarity_score=score,
                state=metadata["state"],
                created_at=metadata["created_at"],
                updated_at=metadata["updated_at"],
                labels=metadata.get("labels", [])
            )
            similar_issues.append(similar)
        
        # Best match is the highest scoring issue
        best_match = similar_issues[0]
        
        # Determine if it's a duplicate
        is_duplicate = best_match.similarity_score >= self.threshold
        
        # Log the decision
        if is_duplicate:
            logger.info(
                f"Found duplicate: Issue matches #{best_match.issue_number} "
                f"with {best_match.similarity_score:.2%} similarity"
            )
        else:
            logger.info(
                f"No duplicate found. Best match: #{best_match.issue_number} "
                f"with {best_match.similarity_score:.2%} similarity"
            )
        
        return DuplicationResult(
            is_duplicate=is_duplicate,
            similar_issues=similar_issues,
            best_match=best_match,
            confidence=best_match.similarity_score
        )
    
    def find_similar_issues(
        self,
        text: str,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SimilarIssue]:
        """
        Find issues similar to the given text.
        
        Args:
            text: Text to search for
            k: Number of similar issues to return
            threshold: Optional similarity threshold
            
        Returns:
            List of similar issues
        """
        if self.vector_store.size() == 0:
            return []
        
        # Generate embedding
        embedding = self.provider.embed_text(text)
        
        # Search
        results = self.vector_store.search(
            query_embedding=embedding,
            k=k,
            threshold=threshold or self.threshold
        )
        
        # Convert to SimilarIssue objects
        similar_issues = []
        for score, metadata in results:
            similar = SimilarIssue(
                issue_number=metadata["number"],
                title=metadata["title"],
                url=metadata["url"],
                similarity_score=score,
                state=metadata["state"],
                created_at=metadata["created_at"],
                updated_at=metadata["updated_at"],
                labels=metadata.get("labels", [])
            )
            similar_issues.append(similar)
        
        return similar_issues
    
    def update_index(
        self,
        owner: str,
        repo: str,
        force: bool = False
    ) -> int:
        """
        Update the index with new issues.
        
        Args:
            owner: Repository owner
            repo: Repository name
            force: Force full reindex
            
        Returns:
            Number of new issues indexed
        """
        if force or self._should_reindex():
            # Clear and rebuild
            self.vector_store.clear()
            self.indexed_issues.clear()
            return self.index_existing_issues(owner, repo)
        
        # Incremental update - get issues updated since last index
        if self.last_index_time:
            since = self.last_index_time.isoformat()
            logger.info(f"Updating index with issues modified since {since}")
            
            # This would need GitHub API support for "since" parameter
            # For now, just reindex
            return self.index_existing_issues(owner, repo)
        
        return 0
    
    def _should_reindex(self) -> bool:
        """Check if we should reindex based on cache age."""
        if not self.last_index_time:
            return True
        
        age = datetime.now() - self.last_index_time
        return age > self.cache_ttl
    
    def _get_cache_file(self, owner: str, repo: str) -> Path:
        """Get cache file path for a repository."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{owner}_{repo}_embeddings.json"
    
    def _save_cache(self, cache_file: Path) -> None:
        """Save indexed issues to cache."""
        try:
            # Save vector store
            vector_file = cache_file.with_suffix(".vectors")
            self.vector_store.save(str(vector_file))
            
            # Save metadata
            cache_data = {
                "indexed_issues": self.indexed_issues,
                "last_index_time": self.last_index_time.isoformat() if self.last_index_time else None,
                "vector_store_size": self.vector_store.size()
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved cache to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self, cache_file: Path) -> bool:
        """Load indexed issues from cache."""
        try:
            if not cache_file.exists():
                return False
            
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > self.cache_ttl:
                logger.info("Cache expired, will reindex")
                return False
            
            # Load metadata
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            self.indexed_issues = cache_data["indexed_issues"]
            
            # Convert issue numbers to int (JSON keys are strings)
            self.indexed_issues = {
                int(k): v for k, v in self.indexed_issues.items()
            }
            
            if cache_data.get("last_index_time"):
                self.last_index_time = datetime.fromisoformat(cache_data["last_index_time"])
            
            # Load vector store
            vector_file = cache_file.with_suffix(".vectors")
            if vector_file.exists():
                self.vector_store.load(str(vector_file))
                logger.debug(f"Loaded {self.vector_store.size()} vectors from cache")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.vector_store.clear()
        self.indexed_issues.clear()
        self.last_index_time = None
        
        # Remove cache files
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*_embeddings.json"):
                cache_file.unlink(missing_ok=True)
                
                vector_file = cache_file.with_suffix(".vectors")
                vector_file.unlink(missing_ok=True)
        
        logger.info("Cleared all cached data")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the deduplicator."""
        return {
            "indexed_issues": len(self.indexed_issues),
            "vector_store_size": self.vector_store.size(),
            "last_index_time": self.last_index_time.isoformat() if self.last_index_time else None,
            "cache_dir": str(self.cache_dir),
            "threshold": self.threshold,
            "provider": self.provider.__class__.__name__
        }


def create_embedding_deduplicator(
    config: Any,
    github_client: GitHubClient
) -> EmbeddingDeduplicator:
    """
    Factory function to create an embedding deduplicator.
    
    Args:
        config: Configuration object
        github_client: GitHub client
        
    Returns:
        Configured EmbeddingDeduplicator
    """
    # Create embedding provider
    provider = create_embedding_provider(config)
    
    # Create deduplicator
    return EmbeddingDeduplicator(
        embedding_provider=provider,
        github_client=github_client,
        similarity_threshold=config.deduplication.embedding.similarity_threshold,
        vector_store_type=config.deduplication.embedding.vector_store_type,
        cache_dir=Path(config.deduplication.embedding.cache_path),
        cache_ttl_hours=24  # Could be configurable
    )