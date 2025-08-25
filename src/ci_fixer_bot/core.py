"""
Core ci_fixer_bot functionality.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console

from .config import Config
from .github_client import GitHubClient
from .llm_providers import create_llm_provider
from .models import AnalysisResults, CIFailure, Issue

console = Console()


@dataclass
class CIFixerBot:
    """Main ci_fixer_bot class that orchestrates the analysis process."""
    
    config: Config
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize components after dataclass creation."""
        self.github_client = GitHubClient(token=self.config.github_token)
        self.llm_provider = create_llm_provider(self.config.llm)
    
    def analyze_repository(
        self,
        repository_url: str,
        analyze_runs: int = 5,
        focus: str = "all",
        min_priority: int = 1,
        dry_run: bool = False,
    ) -> AnalysisResults:
        """
        Main entry point for analyzing a repository's CI failures.
        
        Args:
            repository_url: GitHub repository URL
            analyze_runs: Number of recent CI runs to analyze
            focus: Type of issues to focus on (safe, tactical, strategic, all)
            min_priority: Minimum priority threshold (1-100)
            dry_run: If True, don't create actual GitHub issues
            
        Returns:
            AnalysisResults containing issues and statistics
        """
        start_time = time.time()
        
        if self.verbose:
            console.print(f"🔍 Analyzing {repository_url}...")
        
        # Step 1: Extract owner/repo from URL
        owner, repo = self._parse_repo_url(repository_url)
        
        # Step 2: Fetch CI failures
        if self.verbose:
            console.print(f"📥 Fetching last {analyze_runs} CI runs...")
        
        failures = self._fetch_ci_failures(owner, repo, analyze_runs)
        
        if not failures:
            if self.verbose:
                console.print("✅ No CI failures found!")
            return AnalysisResults(
                repository_url=repository_url,
                issues=[],
                stats={"analysis_time_seconds": time.time() - start_time}
            )
        
        if self.verbose:
            console.print(f"🔍 Found {len(failures)} failure patterns")
        
        # Step 3: Analyze failures and generate issues
        if self.verbose:
            console.print("🧠 Analyzing failures with AI...")
        
        issues = self._analyze_failures(failures, owner, repo)
        
        # Step 4: Filter issues based on criteria
        filtered_issues = self._filter_issues(issues, focus, min_priority)
        
        if self.verbose:
            console.print(f"📋 Generated {len(filtered_issues)} actionable issues")
        
        # Step 5: Check for duplicates and create GitHub issues (unless dry run)
        if not dry_run:
            if self.verbose:
                console.print("🔍 Checking for duplicate issues...")
            
            filtered_issues = self._check_and_create_issues(filtered_issues, owner, repo)
            
        else:
            if self.verbose:
                console.print("🔍 Dry run - checking for potential duplicates...")
                self._check_and_create_issues(filtered_issues, owner, repo, dry_run=True)
        
        # Compile results
        results = AnalysisResults(
            repository_url=repository_url,
            issues=filtered_issues,
            stats={
                "ci_runs_analyzed": analyze_runs,
                "total_failures": len(failures),
                "patterns_found": len(issues),
                "issues_created": len(filtered_issues),
                "analysis_time_seconds": time.time() - start_time,
            }
        )
        
        return results
    
    def _parse_repo_url(self, url: str) -> tuple[str, str]:
        """Extract owner and repo name from GitHub URL."""
        # Handle various GitHub URL formats
        url = url.rstrip('/')
        
        if url.startswith('https://github.com/'):
            parts = url.split('/')
            return parts[-2], parts[-1].replace('.git', '')
        elif url.startswith('git@github.com:'):
            repo_part = url.split(':')[1]
            parts = repo_part.split('/')
            return parts[0], parts[1].replace('.git', '')
        else:
            raise ValueError(f"Invalid GitHub URL format: {url}")
    
    def _fetch_ci_failures(self, owner: str, repo: str, analyze_runs: int) -> List[CIFailure]:
        """Fetch and parse CI failures from GitHub Actions."""
        try:
            # Get recent workflow runs
            runs = self.github_client.get_workflow_runs(
                owner=owner,
                repo=repo,
                status="failure",
                per_page=analyze_runs
            )
            
            failures = []
            
            for run in runs:
                # Get jobs for this run
                jobs = self.github_client.get_workflow_jobs(
                    owner=owner,
                    repo=repo,
                    run_id=run["id"]
                )
                
                for job in jobs:
                    if job["conclusion"] == "failure":
                        # Get logs for failed job
                        try:
                            logs = self.github_client.get_job_logs(
                                owner=owner,
                                repo=repo,
                                job_id=job["id"]
                            )
                            
                            failure = CIFailure(
                                run_id=run["id"],
                                job_name=job["name"],
                                workflow_name=run["name"],
                                created_at=run["created_at"],
                                logs=logs,
                                run_url=run["html_url"]
                            )
                            failures.append(failure)
                            
                        except Exception as e:
                            if self.verbose:
                                console.print(f"⚠️  Could not fetch logs for job {job['name']}: {e}")
                            continue
            
            return failures
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch CI failures: {e}")
    
    def _analyze_failures(self, failures: List[CIFailure], owner: str, repo: str) -> List[Issue]:
        """Analyze failures using LLM and create issues."""
        from .analyzers import FailureAnalyzer
        
        analyzer = FailureAnalyzer(
            llm_provider=self.llm_provider,
            config=self.config,
            github_client=self.github_client,
            owner=owner,
            repo=repo
        )
        
        # Store analyzer reference for deduplication
        self._current_analyzer = analyzer
        
        return analyzer.analyze_failures(failures)
    
    def _filter_issues(self, issues: List[Issue], focus: str, min_priority: int) -> List[Issue]:
        """Filter issues based on focus and priority criteria."""
        filtered = []
        
        for issue in issues:
            # Filter by focus
            if focus != "all" and issue.risk_level != focus:
                continue
            
            # Filter by priority
            if issue.priority < min_priority:
                continue
            
            filtered.append(issue)
        
        return filtered
    
    def _check_and_create_issues(
        self, 
        issues: List[Issue], 
        owner: str, 
        repo: str,
        dry_run: bool = False
    ) -> List[Issue]:
        """Check for duplicates and create issues, handling deduplication."""
        from .deduplication import IssueDeduplicator
        
        # Check if deduplication is enabled
        if not self.config.deduplication.enabled:
            # Deduplication disabled, create all issues directly
            if not dry_run:
                for issue in issues:
                    if self.verbose:
                        console.print(f"✅ Creating issue: {issue.title}")
                    
                    github_issue = self.github_client.create_issue(
                        owner=owner,
                        repo=repo,
                        title=issue.title,
                        body=issue.body,
                        labels=issue.labels,
                        assignees=issue.assignees,
                    )
                    issue.github_url = github_issue["html_url"]
                    issue.issue_number = github_issue["number"]
            return issues
        
        # Initialize deduplicator
        deduplicator = IssueDeduplicator(
            github_client=self.github_client,
            similarity_threshold=self.config.deduplication.similarity_threshold,
            exact_match_threshold=self.config.deduplication.exact_match_threshold
        )
        
        # Get analyzer to access failure contexts
        analyzer = self._get_current_analyzer()
        
        created_issues = []
        skipped_count = 0
        
        for issue in issues:
            try:
                # Get failure context for this issue
                failure_context = analyzer.get_failure_context(issue) if analyzer else {}
                
                # Check for duplicates
                dup_result = deduplicator.check_for_duplicate(owner, repo, issue, failure_context)
                
                if dup_result.is_duplicate:
                    if self.verbose:
                        console.print(f"⚠️  Skipping duplicate issue: {issue.title}")
                        console.print(f"   Similar to: {dup_result.existing_issue.get('title', 'Unknown')}")
                        console.print(f"   Similarity: {dup_result.similarity_score:.2f}")
                    
                    # Optionally add a comment to the existing issue
                    if self.config.deduplication.update_existing and not dry_run:
                        self._update_existing_issue(owner, repo, dup_result.existing_issue, issue)
                    
                    skipped_count += 1
                    continue
                
                # Not a duplicate, create the issue
                if not dry_run:
                    if self.verbose:
                        console.print(f"✅ Creating issue: {issue.title}")
                    
                    github_issue = self.github_client.create_issue(
                        owner=owner,
                        repo=repo,
                        title=issue.title,
                        body=issue.body,
                        labels=issue.labels,
                        assignees=issue.assignees,
                    )
                    issue.github_url = github_issue["html_url"]
                    issue.issue_number = github_issue["number"]
                else:
                    if self.verbose:
                        console.print(f"✅ Would create: {issue.title}")
                
                created_issues.append(issue)
                
            except Exception as e:
                if self.verbose:
                    console.print(f"❌ Error processing issue '{issue.title}': {e}")
                # Continue with other issues even if one fails
                continue
        
        if self.verbose and skipped_count > 0:
            console.print(f"📊 Skipped {skipped_count} duplicate issues")
            console.print(f"📊 {'Would create' if dry_run else 'Created'} {len(created_issues)} new issues")
        
        return created_issues
    
    def _get_current_analyzer(self):
        """Get the current analyzer instance. This is a bit of a hack but needed for context."""
        # In a real implementation, you'd pass the analyzer instance or refactor the architecture
        # For now, we'll store it as an instance variable when we create it
        return getattr(self, '_current_analyzer', None)
    
    def _update_existing_issue(self, owner: str, repo: str, existing_issue: dict, new_issue: Issue):
        """Add a comment to existing issue noting the new occurrence."""
        comment_body = (
            f"## 🔄 Additional CI Failure Detected\n\n"
            f"A similar CI failure pattern was detected and would have generated this issue:\n\n"
            f"**{new_issue.title}**\n\n"
            f"This suggests the issue is still occurring. Please review the current status.\n\n"
            f"---\n"
            f"*🤖 Generated by ci_fixer_bot - AI-powered CI issue analysis*"
        )
        
        try:
            self.github_client.add_issue_comment(
                owner=owner,
                repo=repo,
                issue_number=existing_issue['number'],
                body=comment_body
            )
            
            if self.verbose:
                console.print(f"💬 Added update comment to issue #{existing_issue['number']}")
                
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️  Could not update existing issue: {e}")