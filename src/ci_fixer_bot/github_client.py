"""
GitHub API client for ci_fixer_bot.
"""

import os
import subprocess
from typing import Dict, List, Optional

import requests


class GitHubClient:
    """Client for interacting with GitHub API."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token. If None, will try to use
                   gh CLI authentication or GITHUB_TOKEN env var.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        
        # If no token provided, try to use gh CLI
        if not self.token:
            try:
                result = subprocess.run(
                    ["gh", "auth", "token"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.token = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        if not self.token:
            raise ValueError(
                "GitHub authentication required. Either:\n"
                "1. Set GITHUB_TOKEN environment variable\n"
                "2. Pass token parameter\n"
                "3. Authenticate with 'gh auth login'"
            )
        
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ci_fixer_bot/0.1.0"
        }
    
    def get_workflow_runs(
        self, 
        owner: str, 
        repo: str, 
        status: str = "failure",
        per_page: int = 30
    ) -> List[Dict]:
        """
        Get workflow runs for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            status: Run status (completed, failure, success, etc.)
            per_page: Number of results per page
            
        Returns:
            List of workflow run objects
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs"
        params = {
            "status": status,
            "per_page": per_page
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()["workflow_runs"]
    
    def get_workflow_jobs(self, owner: str, repo: str, run_id: int) -> List[Dict]:
        """
        Get jobs for a specific workflow run.
        
        Args:
            owner: Repository owner
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            List of job objects
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()["jobs"]
    
    def get_job_logs(self, owner: str, repo: str, job_id: int) -> str:
        """
        Get logs for a specific job.
        
        Args:
            owner: Repository owner
            repo: Repository name
            job_id: Job ID
            
        Returns:
            Job logs as string
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.text
    
    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body (markdown)
            labels: List of label names
            assignees: List of assignee usernames
            
        Returns:
            Created issue object
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        
        data = {
            "title": title,
            "body": body
        }
        
        if labels:
            data["labels"] = labels
        
        if assignees:
            data["assignees"] = assignees
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def search_issues(
        self,
        owner: str,
        repo: str,
        query: str,
        state: str = "open"
    ) -> List[Dict]:
        """
        Search for issues in a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            query: Search query
            state: Issue state (open, closed, all)
            
        Returns:
            List of issue objects
        """
        search_query = f"repo:{owner}/{repo} {query} state:{state}"
        url = f"{self.base_url}/search/issues"
        
        params = {"q": search_query}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()["items"]
    
    def get_repository_info(self, owner: str, repo: str) -> Dict:
        """
        Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository object
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """
        Get content of a file from the repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Git reference (branch, tag, commit)
            
        Returns:
            File content as string
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 404:
            raise FileNotFoundError(f"File not found: {path}")
        
        response.raise_for_status()
        
        import base64
        content = response.json()["content"]
        return base64.b64decode(content).decode("utf-8")
    
    def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        state: Optional[str] = None
    ) -> Dict:
        """
        Update an existing GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            title: New title (optional)
            body: New body (optional)
            labels: New labels (optional)
            assignees: New assignees (optional)
            state: New state - open or closed (optional)
            
        Returns:
            Updated issue object
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}"
        
        data = {}
        
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees
        if state is not None:
            data["state"] = state
        
        response = requests.patch(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict:
        """
        Add a comment to an issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            body: Comment body (markdown)
            
        Returns:
            Comment object
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        
        data = {"body": body}
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()