"""
Data models for ci_fixer_bot.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CIFailure:
    """Represents a CI failure from a workflow run."""
    run_id: int
    job_name: str
    workflow_name: str
    created_at: str
    logs: str
    run_url: str
    
    # Analysis fields (populated later)
    error_patterns: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    failure_type: Optional[str] = None


@dataclass
class Issue:
    """Represents a GitHub issue to be created."""
    title: str
    body: str
    risk_level: str  # safe, tactical, strategic
    priority: int  # 1-100
    estimated_effort: str  # e.g., "30 minutes", "2-4 hours"
    required_expertise: str  # junior, mid, senior
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    
    # Populated after creation
    github_url: Optional[str] = None
    issue_number: Optional[int] = None


@dataclass
class AnalysisResults:
    """Results of analyzing a repository."""
    repository_url: str
    issues: List[Issue]
    stats: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "repository_url": self.repository_url,
            "issues": [
                {
                    "title": issue.title,
                    "risk_level": issue.risk_level,
                    "priority": issue.priority,
                    "estimated_effort": issue.estimated_effort,
                    "required_expertise": issue.required_expertise,
                    "labels": issue.labels,
                    "assignees": issue.assignees,
                    "github_url": issue.github_url,
                }
                for issue in self.issues
            ],
            "stats": self.stats,
        }


@dataclass
class ErrorPattern:
    """Represents a detected error pattern across multiple failures."""
    pattern_type: str  # e.g., "flaky_test", "linting", "dependency_issue"
    error_signature: str  # Unique identifier for this error
    occurrences: int
    affected_runs: List[int]
    first_seen: datetime
    last_seen: datetime
    
    # Analysis
    root_cause: Optional[str] = None
    suggested_fix: Optional[str] = None
    risk_level: Optional[str] = None


@dataclass
class RiskAssessment:
    """Risk assessment for a particular failure or fix."""
    risk_level: str  # safe, tactical, strategic
    blast_radius: str  # Description of what could break
    required_expertise: str  # junior, mid, senior
    estimated_effort: str  # Time estimate
    confidence: float  # 0.0 - 1.0, how confident we are in this assessment
    
    reasoning: str = ""  # Why this risk level was assigned


@dataclass
class DuplicationResult:
    """Result of checking if an issue is a duplicate."""
    is_duplicate: bool
    similar_issues: List[Any]  # List of SimilarIssue objects
    best_match: Optional[Any] = None  # Best matching SimilarIssue
    confidence: float = 0.0  # Confidence score (0.0-1.0)