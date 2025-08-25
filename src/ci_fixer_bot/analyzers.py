"""
Failure analysis engine for ci_fixer_bot.
"""

import json
import re
from typing import Dict, List, Optional, Set

from .config import Config
from .github_client import GitHubClient
from .llm_providers import LLMProvider
from .models import CIFailure, Issue, RiskAssessment
from .risk_assessor import RiskAssessor


class FailureAnalyzer:
    """Analyzes CI failures and generates actionable issues."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Config,
        github_client: GitHubClient,
        owner: str,
        repo: str
    ):
        self.llm_provider = llm_provider
        self.config = config
        self.github_client = github_client
        self.owner = owner
        self.repo = repo
        self.risk_assessor = RiskAssessor(config)
        self.failure_contexts = {}  # Store failure context for deduplication
    
    def analyze_failures(self, failures: List[CIFailure]) -> List[Issue]:
        """
        Analyze CI failures and generate issues.
        
        Args:
            failures: List of CI failures to analyze
            
        Returns:
            List of issues to create
        """
        if not failures:
            return []
        
        # Step 1: Parse and categorize failures
        parsed_failures = self._parse_failures(failures)
        
        # Step 2: Group similar failures
        grouped_failures = self._group_failures(parsed_failures)
        
        # Step 3: Generate issues using LLM
        issues = self._generate_issues(grouped_failures)
        
        # Step 4: Enhance issues with risk assessment
        enhanced_issues = self._enhance_issues(issues)
        
        return enhanced_issues
    
    def _parse_failures(self, failures: List[CIFailure]) -> List[CIFailure]:
        """Parse failure logs to extract structured information."""
        
        for failure in failures:
            # Extract error patterns
            failure.error_patterns = self._extract_error_patterns(failure.logs)
            
            # Extract affected files
            failure.affected_files = self._extract_affected_files(failure.logs)
            
            # Classify failure type
            failure.failure_type = self._classify_failure_type(failure)
        
        return failures
    
    def _extract_error_patterns(self, logs: str) -> List[str]:
        """Extract common error patterns from logs."""
        patterns = []
        
        # Common error patterns
        error_regexes = [
            # Test failures
            r"FAILED.*::(.*)",
            r"AssertionError: (.*)",
            r"Error: (.*) at line (\d+)",
            
            # Linting errors
            r"(.*\.py):\d+:\d+: (.*)",
            r"(.*\.js):\d+:\d+\s+(.*)",
            
            # Build errors
            r"Error: Cannot find module '(.*)'",
            r"ModuleNotFoundError: No module named '(.*)'",
            r"could not connect to server: (.*)",
            
            # Security issues
            r"(High|Critical) severity vulnerability",
            r"CVE-\d{4}-\d{4,}",
        ]
        
        for regex in error_regexes:
            matches = re.findall(regex, logs, re.IGNORECASE | re.MULTILINE)
            if matches:
                patterns.extend([str(match) for match in matches])
        
        # Deduplicate and return first 10
        return list(set(patterns))[:10]
    
    def _extract_affected_files(self, logs: str) -> List[str]:
        """Extract file paths mentioned in error logs."""
        files = set()
        
        # File path patterns
        file_regexes = [
            r"([^/\s]+(?:/[^/\s]+)*\.\w+):\d+",  # file.ext:line
            r"at ([^/\s]+(?:/[^/\s]+)*\.\w+):\d+:\d+",  # at file.ext:line:col
            r"File \"([^\"]+)\"",  # File "path"
            r"in file ([^\s]+\.\w+)",  # in file path.ext
        ]
        
        for regex in file_regexes:
            matches = re.findall(regex, logs)
            files.update(matches)
        
        # Filter out common noise
        noise_patterns = ["node_modules/", "/tmp/", "/__pycache__/", ".git/"]
        
        filtered_files = []
        for file in files:
            if not any(noise in file for noise in noise_patterns):
                filtered_files.append(file)
        
        return filtered_files[:20]  # Limit to prevent noise
    
    def _classify_failure_type(self, failure: CIFailure) -> str:
        """Classify the type of failure based on patterns."""
        logs_lower = failure.logs.lower()
        
        # Classification rules
        if any(pattern in logs_lower for pattern in ["trailing whitespace", "quote style", "semicolon"]):
            return "linting"
        elif any(pattern in logs_lower for pattern in ["test failed", "assertion error", "spec failed"]):
            return "test_failure"
        elif any(pattern in logs_lower for pattern in ["could not connect", "connection refused", "timeout"]):
            return "infrastructure"
        elif any(pattern in logs_lower for pattern in ["vulnerability", "security", "cve-"]):
            return "security"
        elif any(pattern in logs_lower for pattern in ["module not found", "import error", "dependency"]):
            return "dependency"
        elif any(pattern in logs_lower for pattern in ["database", "migration", "sql"]):
            return "database"
        else:
            return "unknown"
    
    def _group_failures(self, failures: List[CIFailure]) -> Dict[str, List[CIFailure]]:
        """Group similar failures together."""
        groups = {}
        
        for failure in failures:
            # Create a key based on failure type and main error pattern
            key_parts = [failure.failure_type]
            
            if failure.error_patterns:
                # Use first error pattern as part of key
                key_parts.append(failure.error_patterns[0][:50])
            
            if failure.affected_files:
                # Use first affected file as part of key
                key_parts.append(failure.affected_files[0])
            
            key = "|".join(key_parts)
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(failure)
        
        return groups
    
    def _generate_issues(self, grouped_failures: Dict[str, List[CIFailure]]) -> List[Issue]:
        """Generate issues using LLM analysis."""
        issues = []
        
        for group_key, group_failures in grouped_failures.items():
            try:
                # Prepare context for LLM
                context = self._build_analysis_context(group_failures)
                
                # Generate issue using LLM
                issue = self._generate_single_issue(context, group_failures)
                
                if issue:
                    # Store failure context for deduplication
                    self._store_failure_context(issue, group_failures)
                    issues.append(issue)
                    
            except Exception as e:
                # Fallback to pattern-based issue generation
                fallback_issue = self._generate_fallback_issue(group_failures)
                if fallback_issue:
                    self._store_failure_context(fallback_issue, group_failures)
                    issues.append(fallback_issue)
        
        return issues
    
    def _build_analysis_context(self, failures: List[CIFailure]) -> str:
        """Build context for LLM analysis."""
        context_parts = []
        
        context_parts.append(f"Repository: {self.owner}/{self.repo}")
        context_parts.append(f"Number of similar failures: {len(failures)}")
        
        # Aggregate information
        all_error_patterns = []
        all_affected_files = []
        failure_types = set()
        
        for failure in failures:
            all_error_patterns.extend(failure.error_patterns)
            all_affected_files.extend(failure.affected_files)
            if failure.failure_type:
                failure_types.add(failure.failure_type)
        
        context_parts.append(f"Failure types: {', '.join(failure_types)}")
        
        if all_error_patterns:
            unique_patterns = list(set(all_error_patterns))[:5]
            context_parts.append(f"Common error patterns:\n" + "\n".join(f"  - {p}" for p in unique_patterns))
        
        if all_affected_files:
            unique_files = list(set(all_affected_files))[:10]
            context_parts.append(f"Affected files:\n" + "\n".join(f"  - {f}" for f in unique_files))
        
        # Include sample logs (truncated)
        sample_failure = failures[0]
        truncated_logs = sample_failure.logs[:2000] + "..." if len(sample_failure.logs) > 2000 else sample_failure.logs
        context_parts.append(f"Sample logs:\n```\n{truncated_logs}\n```")
        
        return "\n\n".join(context_parts)
    
    def _generate_single_issue(self, context: str, failures: List[CIFailure]) -> Optional[Issue]:
        """Generate a single issue using LLM."""
        
        prompt = f"""
Analyze this CI failure and create a GitHub issue following senior engineer best practices.

{context}

Please provide your analysis in JSON format:

{{
  "title": "Clear, actionable title (50-80 chars)",
  "risk_level": "safe|tactical|strategic", 
  "priority": 1-100,
  "estimated_effort": "e.g. '30 minutes', '2-4 hours'",
  "required_expertise": "junior|mid|senior",
  "root_cause": "What's actually causing this failure",
  "fix_approach": "Step-by-step approach to fix",
  "why_this_risk_level": "Reasoning for risk assessment",
  "commands_to_run": ["list", "of", "specific", "commands"],
  "related_files": ["files", "that", "need", "attention"]
}}

Risk level guidelines:
- SAFE: Linting, formatting, documentation - no runtime impact
- TACTICAL: Tests, minor updates - needs verification but low blast radius  
- STRATEGIC: Database, auth, payments, security - requires planning

Focus on being specific and actionable. Senior engineers want to know exactly what to do and why.
"""
        
        try:
            llm_response = self.llm_provider.analyze(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                issue_data = json.loads(json_match.group())
                
                # Build issue body
                body = self._build_issue_body(issue_data, failures)
                
                # Create issue object
                issue = Issue(
                    title=f"{self._get_risk_emoji(issue_data['risk_level'])} [{issue_data['risk_level'].upper()}] {issue_data['title']}",
                    body=body,
                    risk_level=issue_data['risk_level'],
                    priority=issue_data['priority'],
                    estimated_effort=issue_data['estimated_effort'],
                    required_expertise=issue_data['required_expertise'],
                    labels=self._generate_labels(issue_data, failures),
                )
                
                return issue
                
        except (json.JSONDecodeError, KeyError) as e:
            # LLM didn't return valid JSON, fall back to pattern matching
            pass
        
        return None
    
    def _build_issue_body(self, issue_data: dict, failures: List[CIFailure]) -> str:
        """Build GitHub issue body from LLM analysis."""
        
        risk_emoji = self._get_risk_emoji(issue_data['risk_level'])
        
        body_parts = [
            f"**Risk Level**: {issue_data['risk_level'].upper()} - {issue_data.get('why_this_risk_level', 'No reasoning provided')}",
            f"**Effort**: {issue_data['estimated_effort']}",
            f"**Required Expertise**: {issue_data['required_expertise']}",
            f"**Blocking**: {len(failures)} recent CI run{'s' if len(failures) != 1 else ''}",
        ]
        
        if issue_data['risk_level'] == 'strategic':
            body_parts.append("\n## ⚠️ HIGH RISK - Review Required\n")
            body_parts.append("This issue requires careful planning and senior engineer review before implementation.")
        
        # Root cause section
        if issue_data.get('root_cause'):
            body_parts.extend([
                "\n## Root Cause",
                issue_data['root_cause']
            ])
        
        # Fix approach
        if issue_data.get('fix_approach'):
            body_parts.extend([
                "\n## Recommended Fix",
                issue_data['fix_approach']
            ])
        
        # Commands to run
        if issue_data.get('commands_to_run'):
            body_parts.append("\n## Commands to Run")
            body_parts.append("```bash")
            for cmd in issue_data['commands_to_run']:
                body_parts.append(cmd)
            body_parts.append("```")
        
        # Related files
        if issue_data.get('related_files'):
            body_parts.extend([
                "\n## Related Files",
                "\n".join(f"- `{file}`" for file in issue_data['related_files'])
            ])
        
        # Recent occurrences
        body_parts.extend([
            "\n## Recent Occurrences",
            f"- Failed in {len(failures)} recent CI runs",
            f"- Most recent: {failures[0].created_at}",
        ])
        
        if len(failures) > 1:
            body_parts.append(f"- First occurrence: {failures[-1].created_at}")
        
        # Links to CI runs
        body_parts.append("\n## CI Run Links")
        for i, failure in enumerate(failures[:3]):  # Limit to 3 links
            body_parts.append(f"- [{failure.workflow_name} - {failure.job_name}]({failure.run_url})")
        
        if len(failures) > 3:
            body_parts.append(f"- ... and {len(failures) - 3} more")
        
        # Footer
        body_parts.extend([
            "\n---",
            "*🤖 Generated by ci_fixer_bot - AI-powered CI issue analysis*"
        ])
        
        return "\n".join(body_parts)
    
    def _generate_fallback_issue(self, failures: List[CIFailure]) -> Optional[Issue]:
        """Generate issue using pattern matching when LLM fails."""
        
        failure_type = failures[0].failure_type
        
        # Fallback templates
        templates = {
            "linting": {
                "title": "Fix code style violations",
                "risk_level": "safe",
                "priority": 30,
                "effort": "15-30 minutes"
            },
            "test_failure": {
                "title": "Fix failing tests",
                "risk_level": "tactical", 
                "priority": 60,
                "effort": "1-2 hours"
            },
            "security": {
                "title": "Address security vulnerabilities",
                "risk_level": "strategic",
                "priority": 90,
                "effort": "2-4 hours"
            },
            "database": {
                "title": "Fix database configuration issue",
                "risk_level": "strategic", 
                "priority": 85,
                "effort": "4-6 hours"
            },
        }
        
        template = templates.get(failure_type, {
            "title": "Fix CI failure",
            "risk_level": "tactical",
            "priority": 50,
            "effort": "1-3 hours"
        })
        
        return Issue(
            title=f"{self._get_risk_emoji(template['risk_level'])} [{template['risk_level'].upper()}] {template['title']}",
            body=f"Automated analysis found {len(failures)} similar CI failures.\n\n"
                 f"Failure type: {failure_type}\n"
                 f"Review the CI logs for details.",
            risk_level=template['risk_level'],
            priority=template['priority'],
            estimated_effort=template['effort'],
            required_expertise="mid",
            labels=["ci-fix", f"type-{failure_type}", "auto-generated"],
        )
    
    def _enhance_issues(self, issues: List[Issue]) -> List[Issue]:
        """Enhance issues with risk assessment and team assignments."""
        
        for issue in issues:
            # Apply risk overrides from config
            for override in self.config.risk_overrides:
                # Simple pattern matching - in production would use more sophisticated matching
                if any(override.pattern.replace("*", "") in file 
                       for file in getattr(issue, 'affected_files', [])):
                    issue.risk_level = override.risk_level
            
            # Apply team assignments
            assignees = self._get_assignees_for_issue(issue)
            if assignees:
                issue.assignees = assignees
        
        return issues
    
    def _get_assignees_for_issue(self, issue: Issue) -> List[str]:
        """Determine who should be assigned to this issue."""
        assignees = []
        
        # Strategic issues get assigned to senior team members
        if issue.risk_level == "strategic":
            # In real implementation, would look up actual team members
            assignees.append("senior-engineer")
        
        # Could add more sophisticated assignment logic here
        
        return assignees
    
    def _generate_labels(self, issue_data: dict, failures: List[CIFailure]) -> List[str]:
        """Generate appropriate labels for the issue."""
        labels = ["ci-fix"]
        
        # Risk level label
        labels.append(f"{issue_data['risk_level']}-fix")
        
        # Failure type labels
        failure_types = set(f.failure_type for f in failures if f.failure_type)
        for failure_type in failure_types:
            labels.append(f"type-{failure_type}")
        
        # Priority label
        priority = issue_data['priority']
        if priority >= 80:
            labels.append("priority-high")
        elif priority >= 50:
            labels.append("priority-medium")
        else:
            labels.append("priority-low")
        
        # Good first issue for safe fixes
        if issue_data['risk_level'] == "safe":
            labels.append("good-first-issue")
        
        return labels
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """Get emoji for risk level."""
        emojis = {
            "safe": "🟢",
            "tactical": "🟡", 
            "strategic": "🔴"
        }
        return emojis.get(risk_level, "⚪")
    
    def _store_failure_context(self, issue: Issue, failures: List[CIFailure]) -> None:
        """Store failure context for an issue to enable deduplication."""
        # Aggregate information from failures
        all_error_patterns = []
        all_affected_files = []
        failure_types = set()
        
        for failure in failures:
            all_error_patterns.extend(failure.error_patterns)
            all_affected_files.extend(failure.affected_files)
            if failure.failure_type:
                failure_types.add(failure.failure_type)
        
        # Create unique issue identifier for storage
        issue_id = f"{issue.title}|{issue.risk_level}|{hash(issue.body)}"
        
        self.failure_contexts[issue_id] = {
            'failure_types': list(failure_types),
            'error_patterns': list(set(all_error_patterns)),
            'affected_files': list(set(all_affected_files)),
            'failure_count': len(failures)
        }
    
    def get_failure_context(self, issue: Issue) -> dict:
        """Get stored failure context for an issue."""
        issue_id = f"{issue.title}|{issue.risk_level}|{hash(issue.body)}"
        return self.failure_contexts.get(issue_id, {})