"""
Risk assessment module for ci_fixer_bot.

Evaluates the risk level of CI fixes based on patterns and configuration.
"""

import re
from typing import List, Dict, Set

from .config import Config
from .models import RiskAssessment


class RiskAssessor:
    """Assesses risk levels for CI fixes."""
    
    def __init__(self, config: Config):
        self.config = config
        self._load_risk_patterns()
    
    def _load_risk_patterns(self) -> None:
        """Load risk assessment patterns."""
        
        # SAFE patterns - cosmetic changes, no runtime impact
        self.safe_patterns = {
            "file_patterns": [
                r".*\.md$",  # Documentation
                r".*\.txt$",  # Text files
                r".*\.yml$",  # Config files (mostly)
                r".*\.yaml$",
                r".*\.json$",  # JSON configs
                r".*\.xml$",  # XML configs
                r".*\.toml$",  # TOML configs
                r"README.*",
                r"CHANGELOG.*",
                r"\.github/.*",  # GitHub workflows (careful!)
                r"docs/.*",  # Documentation
                r"test.*\.py$",  # Test files (usually safe)
                r".*_test\.py$",
                r".*\.test\.js$",
                r".*\.spec\.js$",
            ],
            "error_patterns": [
                "trailing whitespace",
                "quote style",
                "semicolon",
                "indentation",
                "import order",
                "unused import",
                "missing docstring",
                "line too long",
                "extra blank line",
                "missing final newline",
                "eslint",
                "flake8",
                "rubocop.*style",
                "prettier",
                "black formatting",
            ],
            "command_patterns": [
                "npm run lint.*fix",
                "eslint.*--fix",
                "black .",
                "isort .",
                "prettier --write",
                "rubocop -a",
                "gofmt",
            ]
        }
        
        # STRATEGIC patterns - high risk, needs planning
        self.strategic_patterns = {
            "file_patterns": [
                r".*migration.*\.py$",  # Database migrations
                r".*migration.*\.rb$",
                r".*migration.*\.sql$",
                r".*migrate.*\.py$",
                r".*auth.*\.py$",  # Authentication
                r".*auth.*\.rb$",
                r".*auth.*\.js$",
                r".*payment.*\.py$",  # Payment systems
                r".*payment.*\.rb$",
                r".*billing.*\.py$",
                r".*security.*\.py$",  # Security
                r".*crypto.*\.py$",  # Cryptography
                r".*admin.*\.py$",  # Admin functionality
                r"Dockerfile$",  # Docker changes
                r"docker-compose.*\.yml$",
                r".*\.env$",  # Environment files
                r".*\.env\..*$",
                r"requirements\.txt$",  # Dependency changes
                r"Gemfile$",
                r"package\.json$",
                r".*\.lock$",  # Lock files
                r".*\.gradle$",  # Build system
                r".*pom\.xml$",
            ],
            "error_patterns": [
                "database.*connection",
                "migration.*failed",
                "authentication.*failed",
                "authorization.*denied",
                "permission.*denied",
                "security.*violation",
                "crypto.*error",
                "ssl.*error",
                "certificate.*error",
                "vulnerability",
                "cve-",
                "critical severity",
                "high severity",
                "payment.*failed",
                "transaction.*failed",
                "deadlock",
                "race condition",
                "memory.*leak",
                "stack overflow",
                "segmentation fault",
            ],
            "keywords": [
                "production",
                "prod",
                "live",
                "customer",
                "user data",
                "personal information",
                "pii",
                "gdpr",
                "hipaa",
                "compliance",
                "audit",
                "backup",
                "recovery",
                "disaster",
            ]
        }
        
        # TACTICAL patterns - medium risk, needs testing
        self.tactical_patterns = {
            "file_patterns": [
                r".*\.py$",  # Source code (default tactical)
                r".*\.rb$",
                r".*\.js$",
                r".*\.ts$",
                r".*\.jsx$",
                r".*\.tsx$",
                r".*\.go$",
                r".*\.java$",
                r".*\.cpp$",
                r".*\.c$",
                r".*\.php$",
                r".*\.scala$",
                r".*\.rs$",  # Rust
            ],
            "error_patterns": [
                "test.*failed",
                "assertion.*error",
                "unit test",
                "integration test",
                "spec failed",
                "timeout",
                "connection.*timeout",
                "404 not found",
                "500 internal server error",
                "module not found",
                "import error",
                "dependency.*missing",
                "version.*mismatch",
                "deprecated",
                "warning",
            ]
        }
    
    def assess_risk(
        self,
        error_patterns: List[str],
        affected_files: List[str],
        failure_logs: str,
        failure_type: str = None
    ) -> RiskAssessment:
        """
        Assess the risk level of fixing this failure.
        
        Args:
            error_patterns: List of error patterns found
            affected_files: List of files involved
            failure_logs: Full failure logs
            failure_type: Type of failure (optional)
            
        Returns:
            RiskAssessment object
        """
        
        # Check for config overrides first
        override_risk = self._check_risk_overrides(affected_files)
        if override_risk:
            return RiskAssessment(
                risk_level=override_risk,
                blast_radius="Overridden by configuration",
                required_expertise="as_configured",
                estimated_effort="as_configured",
                confidence=1.0,
                reasoning="Risk level set by configuration override"
            )
        
        # Calculate risk scores
        safe_score = self._calculate_safe_score(error_patterns, affected_files, failure_logs)
        strategic_score = self._calculate_strategic_score(error_patterns, affected_files, failure_logs)
        
        # Determine risk level
        if strategic_score > 0.3:
            risk_level = "strategic"
            confidence = min(strategic_score, 0.9)
            blast_radius = self._describe_strategic_blast_radius(affected_files, error_patterns)
            required_expertise = "senior"
            estimated_effort = self._estimate_strategic_effort(strategic_score)
        elif safe_score > 0.7:
            risk_level = "safe"
            confidence = safe_score
            blast_radius = "Development environment only"
            required_expertise = "junior"
            estimated_effort = self._estimate_safe_effort(error_patterns)
        else:
            risk_level = "tactical"
            confidence = 0.6
            blast_radius = "Limited to specific functionality"
            required_expertise = "mid"
            estimated_effort = "1-3 hours"
        
        reasoning = self._build_reasoning(risk_level, safe_score, strategic_score, affected_files)
        
        return RiskAssessment(
            risk_level=risk_level,
            blast_radius=blast_radius,
            required_expertise=required_expertise,
            estimated_effort=estimated_effort,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _check_risk_overrides(self, affected_files: List[str]) -> str:
        """Check if any risk overrides apply."""
        for override in self.config.risk_overrides:
            pattern = override.pattern.replace("*", ".*")
            regex = re.compile(pattern, re.IGNORECASE)
            
            for file in affected_files:
                if regex.search(file):
                    return override.risk_level
        
        return None
    
    def _calculate_safe_score(
        self,
        error_patterns: List[str],
        affected_files: List[str],
        logs: str
    ) -> float:
        """Calculate how 'safe' this fix appears to be."""
        score = 0.0
        total_checks = 0
        
        # Check error patterns
        for pattern in error_patterns:
            total_checks += 1
            for safe_pattern in self.safe_patterns["error_patterns"]:
                if re.search(safe_pattern, pattern, re.IGNORECASE):
                    score += 1.0
                    break
        
        # Check file patterns
        for file in affected_files:
            total_checks += 1
            for safe_pattern in self.safe_patterns["file_patterns"]:
                if re.search(safe_pattern, file, re.IGNORECASE):
                    score += 1.0
                    break
        
        # Check for safe commands in logs
        total_checks += 1
        for command_pattern in self.safe_patterns["command_patterns"]:
            if re.search(command_pattern, logs, re.IGNORECASE):
                score += 1.0
                break
        
        return score / max(total_checks, 1)
    
    def _calculate_strategic_score(
        self,
        error_patterns: List[str],
        affected_files: List[str],
        logs: str
    ) -> float:
        """Calculate how 'strategic' (risky) this fix appears to be."""
        score = 0.0
        max_score = 0
        
        # Check error patterns (high weight)
        for pattern in error_patterns:
            max_score += 2
            for strategic_pattern in self.strategic_patterns["error_patterns"]:
                if re.search(strategic_pattern, pattern, re.IGNORECASE):
                    score += 2.0
                    break
        
        # Check file patterns (high weight)  
        for file in affected_files:
            max_score += 2
            for strategic_pattern in self.strategic_patterns["file_patterns"]:
                if re.search(strategic_pattern, file, re.IGNORECASE):
                    score += 2.0
                    break
        
        # Check for strategic keywords in logs (medium weight)
        max_score += 1
        for keyword in self.strategic_patterns["keywords"]:
            if re.search(keyword, logs, re.IGNORECASE):
                score += 0.5  # Multiple keywords can add up
        
        return score / max(max_score, 1) if max_score > 0 else 0.0
    
    def _describe_strategic_blast_radius(
        self,
        affected_files: List[str],
        error_patterns: List[str]
    ) -> str:
        """Describe what could go wrong with strategic changes."""
        
        risks = []
        
        # Database risks
        db_files = [f for f in affected_files if re.search(r"migration|database|db", f, re.IGNORECASE)]
        if db_files:
            risks.append("Could break database schema or data integrity")
        
        # Auth risks
        auth_files = [f for f in affected_files if re.search(r"auth|login|security", f, re.IGNORECASE)]
        if auth_files:
            risks.append("Could lock out users or compromise security")
        
        # Payment risks
        payment_files = [f for f in affected_files if re.search(r"payment|billing|stripe|paypal", f, re.IGNORECASE)]
        if payment_files:
            risks.append("Could affect revenue or payment processing")
        
        # Infrastructure risks
        infra_files = [f for f in affected_files if re.search(r"docker|deploy|config|env", f, re.IGNORECASE)]
        if infra_files:
            risks.append("Could cause deployment or configuration issues")
        
        if not risks:
            risks.append("Could have system-wide impact requiring rollback")
        
        return "; ".join(risks[:2])  # Limit to top 2 risks
    
    def _estimate_strategic_effort(self, strategic_score: float) -> str:
        """Estimate effort for strategic fixes."""
        if strategic_score > 0.8:
            return "1-2 days + planning + testing"
        elif strategic_score > 0.5:
            return "4-8 hours + coordination"
        else:
            return "2-4 hours + review"
    
    def _estimate_safe_effort(self, error_patterns: List[str]) -> str:
        """Estimate effort for safe fixes."""
        # Auto-fixable patterns
        auto_fixable = [
            "trailing whitespace",
            "quote style", 
            "semicolon",
            "import order",
            "formatting"
        ]
        
        for pattern in error_patterns:
            for auto_pattern in auto_fixable:
                if auto_pattern in pattern.lower():
                    return "5-15 minutes (auto-fixable)"
        
        return "15-45 minutes"
    
    def _build_reasoning(
        self,
        risk_level: str,
        safe_score: float,
        strategic_score: float,
        affected_files: List[str]
    ) -> str:
        """Build human-readable reasoning for the risk assessment."""
        
        reasons = []
        
        if risk_level == "safe":
            reasons.append(f"High confidence safe fix (score: {safe_score:.2f})")
            if any(re.search(r"\.(md|txt|json)$", f) for f in affected_files):
                reasons.append("Only affects documentation/config files")
        
        elif risk_level == "strategic":
            reasons.append(f"High risk indicators detected (score: {strategic_score:.2f})")
            if any(re.search(r"migration|database", f) for f in affected_files):
                reasons.append("Involves database changes")
            if any(re.search(r"auth|security", f) for f in affected_files):
                reasons.append("Affects authentication/security")
        
        else:  # tactical
            reasons.append("Medium risk - requires testing but limited blast radius")
            if strategic_score > 0.1:
                reasons.append(f"Some risk indicators present (score: {strategic_score:.2f})")
        
        return "; ".join(reasons)