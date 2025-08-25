"""
Auto-calibration for deduplication thresholds.
Analyzes existing issues to recommend optimal similarity thresholds.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .github_client import GitHubClient
from .embedding_providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """
    Automatically calibrates similarity thresholds based on existing issues.
    """
    
    def __init__(
        self,
        github_client: GitHubClient,
        embedding_provider: EmbeddingProvider
    ):
        self.github_client = github_client
        self.provider = embedding_provider
    
    def calibrate(
        self,
        owner: str,
        repo: str,
        labels: List[str] = None,
        strategy: str = "balanced"
    ) -> Dict[str, float]:
        """
        Calibrate thresholds based on existing issues.
        
        Args:
            owner: Repository owner
            repo: Repository name
            labels: Labels to filter issues
            strategy: "conservative", "balanced", or "aggressive"
            
        Returns:
            Dict with recommended thresholds and confidence metrics
        """
        logger.info(f"Auto-calibrating thresholds for {owner}/{repo}")
        
        # Fetch existing issues
        issues = self.github_client.list_issues(
            owner=owner,
            repo=repo,
            state="all",
            labels=labels
        )
        
        # Filter out PRs
        issues = [i for i in issues if "pull_request" not in i]
        
        if len(issues) < 5:
            logger.warning(f"Only {len(issues)} issues found - using defaults")
            return self._get_defaults(strategy)
        
        # Generate embeddings
        texts = []
        for issue in issues:
            text = self._extract_text(issue)
            texts.append(text)
        
        embeddings = self.provider.embed_batch(texts)
        
        # Calculate similarity distribution
        similar_pairs, different_pairs = self._find_pairs(issues, embeddings)
        
        if not similar_pairs:
            logger.warning("No duplicate pairs found - using defaults")
            return self._get_defaults(strategy)
        
        # Calculate statistics
        similar_scores = [s for _, _, s in similar_pairs]
        different_scores = [s for _, _, s in different_pairs] if different_pairs else []
        
        similar_mean = np.mean(similar_scores)
        similar_std = np.std(similar_scores)
        similar_min = np.min(similar_scores)
        
        if different_scores:
            different_max = np.max(different_scores)
            different_mean = np.mean(different_scores)
        else:
            different_max = 0.5
            different_mean = 0.5
        
        # Calculate thresholds based on strategy
        if strategy == "conservative":
            # High precision: minimize false positives
            threshold = similar_min - 0.02
            confidence = 0.95
        elif strategy == "aggressive":
            # High recall: minimize false negatives  
            threshold = different_max + 0.02
            confidence = 0.90
        else:  # balanced
            # Optimal F1 score
            threshold = self._find_optimal_threshold(
                similar_scores, 
                different_scores
            )
            confidence = 0.93
        
        # Ensure threshold is in reasonable range
        threshold = max(0.6, min(0.95, threshold))
        
        # Calculate separation quality
        separation = similar_mean - different_mean
        quality = "excellent" if separation > 0.4 else "good" if separation > 0.2 else "poor"
        
        return {
            "threshold": threshold,
            "confidence": confidence,
            "strategy": strategy,
            "separation": separation,
            "quality": quality,
            "samples": len(issues),
            "duplicates_found": len(similar_pairs),
            "recommendation": self._get_recommendation(threshold, quality, len(issues))
        }
    
    def _extract_text(self, issue: Dict) -> str:
        """Extract text from issue for embedding."""
        title = issue.get("title", "")
        body = issue.get("body", "")[:500]  # Limit body length
        labels = " ".join([l["name"] for l in issue.get("labels", [])])
        return f"{title} {body} {labels}"
    
    def _find_pairs(
        self, 
        issues: List[Dict], 
        embeddings: np.ndarray
    ) -> Tuple[List, List]:
        """Find similar and different pairs based on title similarity."""
        similar_pairs = []
        different_pairs = []
        
        n = len(issues)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(embeddings[i], embeddings[j])
                
                # Use title similarity as ground truth
                title_sim = self._title_similarity(
                    issues[i]["title"],
                    issues[j]["title"]
                )
                
                if title_sim > 0.7:  # Likely duplicates
                    similar_pairs.append((i, j, similarity))
                else:
                    different_pairs.append((i, j, similarity))
        
        return similar_pairs, different_pairs
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using word overlap."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _find_optimal_threshold(
        self,
        similar_scores: List[float],
        different_scores: List[float]
    ) -> float:
        """Find threshold that maximizes F1 score."""
        if not different_scores:
            # No negative examples, use conservative approach
            return np.percentile(similar_scores, 5)
        
        # Test various thresholds
        thresholds = np.linspace(0.5, 0.95, 20)
        best_f1 = 0
        best_threshold = 0.75
        
        for threshold in thresholds:
            tp = sum(1 for s in similar_scores if s >= threshold)
            fp = sum(1 for s in different_scores if s >= threshold)
            fn = sum(1 for s in similar_scores if s < threshold)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _get_defaults(self, strategy: str) -> Dict[str, float]:
        """Get default thresholds when calibration isn't possible."""
        defaults = {
            "conservative": 0.85,
            "balanced": 0.80,
            "aggressive": 0.75
        }
        
        return {
            "threshold": defaults.get(strategy, 0.80),
            "confidence": 0.5,
            "strategy": strategy,
            "separation": 0,
            "quality": "unknown",
            "samples": 0,
            "duplicates_found": 0,
            "recommendation": "Using defaults - not enough data for calibration"
        }
    
    def _get_recommendation(self, threshold: float, quality: str, samples: int) -> str:
        """Generate recommendation message."""
        if samples < 10:
            return f"⚠️ Only {samples} issues analyzed. Recommend creating more issues before relying on this threshold."
        elif quality == "excellent":
            return f"✅ Excellent separation between duplicates and different issues. Threshold {threshold:.2f} should work well."
        elif quality == "good":
            return f"✓ Good separation detected. Threshold {threshold:.2f} is reasonable but monitor for false positives."
        else:
            return f"⚠️ Poor separation between duplicates and different issues. Consider manual review."


def auto_calibrate_threshold(
    config,
    github_client: GitHubClient,
    owner: str,
    repo: str,
    strategy: str = "balanced"
) -> float:
    """
    Convenience function to auto-calibrate threshold.
    
    Returns the recommended threshold value.
    """
    from .embedding_providers_local import LocalEmbeddingProvider
    
    # Create embedding provider - for now just use local
    # TODO: Use the configured provider
    provider = LocalEmbeddingProvider()
    
    # Create calibrator
    calibrator = ThresholdCalibrator(github_client, provider)
    
    # Run calibration
    result = calibrator.calibrate(
        owner=owner,
        repo=repo,
        labels=["ci-fix"],
        strategy=strategy
    )
    
    logger.info(f"Auto-calibration result: {result}")
    
    return result["threshold"]