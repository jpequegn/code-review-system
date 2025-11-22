"""
Deduplication Service - Identify and deduplicate similar findings

Detects similar findings and groups them to reduce noise, allowing
developers to focus on unique issues rather than repeated variations.
"""

import math
from typing import Optional, List, Dict, Set, Tuple
from difflib import SequenceMatcher
from sqlalchemy.orm import Session

from src.database import Finding


class DeduplicationService:
    """
    Identify and deduplicate similar findings.

    Problem: Code reviews often contain duplicate or near-duplicate findings
    that are essentially the same issue repeated across multiple locations.
    Showing all duplicates clutters the review.

    Solution: Detect similar findings based on title, description, file path,
    and category, group them, and apply a diversity factor that reduces the
    rank of similar suggestions already shown.

    Example:
        dedup = DeduplicationService(db_session)

        # Find similar findings for a given finding
        similar = dedup.find_similar_findings(finding_id, threshold=0.75)

        # Group all findings by similarity
        groups = dedup.group_similar_findings()

        # Calculate diversity factor for a finding relative to others shown
        diversity = dedup.calculate_diversity_factor(finding_id, shown_ids=[1, 2, 3])
    """

    def __init__(self, db: Session, similarity_threshold: float = 0.7):
        """
        Initialize deduplication service.

        Args:
            db: SQLAlchemy database session
            similarity_threshold: Similarity score threshold (0.0-1.0, default: 0.7)
        """
        self.db = db
        self.similarity_threshold = similarity_threshold

    # ============================================================================
    # Similarity Detection
    # ============================================================================

    def calculate_similarity(self, finding1: Finding, finding2: Finding) -> float:
        """
        Calculate similarity between two findings (0.0-1.0).

        Considers:
        - Title similarity (weight: 0.4)
        - Description similarity (weight: 0.3)
        - Category match (weight: 0.2)
        - Severity match (weight: 0.1)

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Similarity score (0.0-1.0)
        """
        scores = {}

        # Title similarity (most important)
        scores["title"] = self._string_similarity(finding1.title, finding2.title)

        # Description similarity
        scores["description"] = self._string_similarity(
            finding1.description or "", finding2.description or ""
        )

        # Category match (exact match = 1.0)
        scores["category"] = 1.0 if finding1.category == finding2.category else 0.0

        # Severity match (exact match = 1.0)
        scores["severity"] = 1.0 if finding1.severity == finding2.severity else 0.0

        # Weighted average
        weights = {"title": 0.4, "description": 0.3, "category": 0.2, "severity": 0.1}
        similarity = sum(scores[key] * weights[key] for key in scores)

        return similarity

    def find_similar_findings(
        self, finding_id: int, threshold: Optional[float] = None
    ) -> List[Tuple[Finding, float]]:
        """
        Find findings similar to a given finding.

        Returns list of (finding, similarity_score) tuples sorted by
        similarity descending, excluding the original finding.

        Args:
            finding_id: Target finding ID
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            List of (finding, similarity) tuples
        """
        if threshold is None:
            threshold = self.similarity_threshold

        target = self.db.query(Finding).filter(Finding.id == finding_id).first()
        if not target:
            return []

        all_findings = self.db.query(Finding).all()
        similar = []

        for finding in all_findings:
            if finding.id == finding_id:
                continue

            similarity = self.calculate_similarity(target, finding)
            if similarity >= threshold:
                similar.append((finding, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    # ============================================================================
    # Grouping
    # ============================================================================

    def group_similar_findings(
        self, threshold: Optional[float] = None
    ) -> List[List[Finding]]:
        """
        Group all findings by similarity using clustering.

        Uses single-linkage clustering: if A is similar to B and B is
        similar to C, they are in the same group even if A and C aren't
        directly similar.

        Args:
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            List of finding groups (each group is a list of findings)
        """
        if threshold is None:
            threshold = self.similarity_threshold

        findings = self.db.query(Finding).all()
        if not findings:
            return []

        # Union-Find structure for clustering
        parent = {f.id: f.id for f in findings}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Link similar findings
        for i, f1 in enumerate(findings):
            for f2 in findings[i + 1 :]:
                if self.calculate_similarity(f1, f2) >= threshold:
                    union(f1.id, f2.id)

        # Group findings by root
        groups_dict = {}
        for finding in findings:
            root = find(finding.id)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(finding)

        return list(groups_dict.values())

    # ============================================================================
    # Diversity Factor
    # ============================================================================

    def calculate_diversity_factor(
        self, finding_id: int, shown_ids: List[int], threshold: Optional[float] = None
    ) -> float:
        """
        Calculate diversity factor for a finding relative to already-shown findings.

        Diversity factor reduces the score of suggestions that are similar to
        ones already shown to the developer, avoiding redundant suggestions.

        Formula:
        - Start with factor = 1.0 (no reduction)
        - For each shown finding with similarity ≥ threshold:
          - Reduce factor by (similarity_score × 0.5)
          - Cap reduction at 50% total

        Args:
            finding_id: Finding to score
            shown_ids: IDs of findings already shown
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            Diversity factor (0.5-1.0, lower = less diverse/redundant)
        """
        if threshold is None:
            threshold = self.similarity_threshold

        finding = self.db.query(Finding).filter(Finding.id == finding_id).first()
        if not finding or not shown_ids:
            return 1.0

        max_similarity = 0.0
        for shown_id in shown_ids:
            shown = self.db.query(Finding).filter(Finding.id == shown_id).first()
            if shown:
                similarity = self.calculate_similarity(finding, shown)
                if similarity >= threshold:
                    max_similarity = max(max_similarity, similarity)

        # Reduction factor: max 50%, scales with similarity
        reduction = max_similarity * 0.5
        factor = 1.0 - reduction
        factor = max(factor, 0.5)  # Cap at 0.5 (can't reduce more than 50%)

        return factor

    def deduplicate_findings(
        self, findings: List[Finding], max_shown: int = 5
    ) -> List[Finding]:
        """
        Select diverse findings to show (avoid duplicates).

        Greedy algorithm: repeatedly select the finding with highest
        diversity factor relative to already-selected findings.

        Args:
            findings: List of findings to deduplicate
            max_shown: Maximum findings to return

        Returns:
            Deduplicated list of findings (up to max_shown)
        """
        if not findings or max_shown <= 0:
            return []

        selected = [findings[0]]
        candidates = findings[1:]

        while candidates and len(selected) < max_shown:
            # Find candidate with highest diversity
            best_candidate = None
            best_diversity = 0.0

            for candidate in candidates:
                diversity = self.calculate_diversity_factor(
                    candidate.id, [f.id for f in selected]
                )
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate is None:
                break

            selected.append(best_candidate)
            candidates.remove(best_candidate)

        return selected

    # ============================================================================
    # Helpers
    # ============================================================================

    def _string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using SequenceMatcher ratio.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0.0-1.0)
        """
        if not str1 or not str2:
            return 0.0 if (str1 and str2) else 1.0

        # Normalize: convert to lowercase
        s1 = str1.lower()
        s2 = str2.lower()

        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()

    def get_deduplication_report(self) -> Dict:
        """
        Generate deduplication analysis report.

        Includes:
        - Total findings
        - Number of groups detected
        - Group sizes distribution
        - Average group size

        Returns:
            Report dict with deduplication statistics
        """
        findings = self.db.query(Finding).all()
        groups = self.group_similar_findings()

        group_sizes = [len(g) for g in groups]

        return {
            "total_findings": len(findings),
            "total_groups": len(groups),
            "singleton_groups": sum(1 for size in group_sizes if size == 1),
            "duplicate_groups": sum(1 for size in group_sizes if size > 1),
            "avg_group_size": sum(group_sizes) / len(groups) if groups else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "groups": [[f.id for f in g] for g in groups],
        }
