"""
Personal Pattern Learning

Learns individual patterns and habits that lead to bugs,
identifying personal mistake patterns and high-risk code locations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class IssueType(str, Enum):
    """Types of issues that can be learned."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    LOGIC = "logic"


class CodeContext(str, Enum):
    """Code locations where issues commonly occur."""

    LOOPS = "loops"
    ASYNC = "async"
    API_HANDLERS = "api_handlers"
    VALIDATION = "validation"
    ERROR_HANDLING = "error_handling"
    CONCURRENCY = "concurrency"
    DATABASE = "database"
    EXTERNAL_CALLS = "external_calls"


@dataclass
class PersonalPattern:
    """Represents a learned personal pattern."""

    pattern_id: str
    issue_type: IssueType
    code_context: CodeContext
    confidence: float  # 0-1, how sure we are of this pattern
    frequency: int  # How many times we've made this mistake
    description: str  # "Off-by-one errors in loop conditions"
    examples: List[str]  # Past files/commits with this issue
    preventions: List[str]  # How to avoid this pattern
    last_occurrence: Optional[str]  # Last commit where this happened


class PatternLearner:
    """Learns personal coding patterns and mistakes."""

    def __init__(self):
        """Initialize pattern learner."""
        self.patterns: Dict[str, PersonalPattern] = {}
        self.issue_counts: Dict[IssueType, int] = {t: 0 for t in IssueType}
        self.context_counts: Dict[CodeContext, int] = {c: 0 for c in CodeContext}

    def record_issue(
        self,
        issue_type: IssueType,
        code_context: CodeContext,
        file_path: str,
        commit_sha: str,
        description: str,
    ) -> None:
        """
        Record an issue occurrence to learn from.

        Args:
            issue_type: Type of issue found
            code_context: Where in code it was found
            file_path: File where issue occurred
            commit_sha: Commit where issue was introduced
            description: Description of the issue
        """
        pattern_id = f"{issue_type}_{code_context}"
        self.issue_counts[issue_type] += 1
        self.context_counts[code_context] += 1

        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_occurrence = commit_sha
            if file_path not in pattern.examples:
                pattern.examples.append(file_path)
        else:
            # Create new pattern
            self.patterns[pattern_id] = PersonalPattern(
                pattern_id=pattern_id,
                issue_type=issue_type,
                code_context=code_context,
                confidence=0.5,  # Start with moderate confidence
                frequency=1,
                description=f"{issue_type.value} issues in {code_context.value}",
                examples=[file_path],
                preventions=self._get_preventions(issue_type, code_context),
                last_occurrence=commit_sha,
            )

        # Boost confidence based on frequency
        if pattern_id in self.patterns:
            self.patterns[pattern_id].confidence = min(
                1.0, 0.5 + (self.patterns[pattern_id].frequency / 20.0)
            )

    def get_patterns_by_type(self, issue_type: IssueType) -> List[PersonalPattern]:
        """
        Get all patterns for a specific issue type.

        Args:
            issue_type: Type of issue

        Returns:
            List of patterns, sorted by frequency
        """
        patterns = [p for p in self.patterns.values() if p.issue_type == issue_type]
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def get_patterns_by_context(self, code_context: CodeContext) -> List[PersonalPattern]:
        """
        Get all patterns for a specific code context.

        Args:
            code_context: Code location type

        Returns:
            List of patterns, sorted by frequency
        """
        patterns = [p for p in self.patterns.values() if p.code_context == code_context]
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def get_high_confidence_patterns(self) -> List[PersonalPattern]:
        """
        Get patterns with high confidence (learned well).

        Returns:
            Patterns with confidence >= 0.7, sorted by frequency
        """
        patterns = [p for p in self.patterns.values() if p.confidence >= 0.7]
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def predict_issue_likelihood(self, code_context: CodeContext) -> Dict[IssueType, float]:
        """
        Predict likelihood of different issue types in given context.

        Args:
            code_context: Code location type

        Returns:
            Dict of issue_type -> probability
        """
        patterns = self.get_patterns_by_context(code_context)

        likelihoods: Dict[IssueType, float] = {t: 0.0 for t in IssueType}

        total_issues = sum(self.issue_counts.values())
        if total_issues == 0:
            return likelihoods

        for pattern in patterns:
            # Likelihood = (frequency * confidence) / total_issues
            likelihoods[pattern.issue_type] += (pattern.frequency * pattern.confidence) / total_issues

        return likelihoods

    def get_weak_spots(self) -> List[tuple]:
        """
        Get your top weak spots (most frequent mistakes).

        Returns:
            List of (pattern, frequency) tuples
        """
        patterns = sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True)
        return [(p, p.frequency) for p in patterns[:5]]

    def get_risk_areas(self) -> Dict[CodeContext, float]:
        """
        Get risk scores for different code contexts.

        Returns:
            Dict of code_context -> risk_score (0-1)
        """
        total_issues = sum(self.issue_counts.values())
        if total_issues == 0:
            return {c: 0.0 for c in CodeContext}

        return {
            context: self.context_counts[context] / total_issues
            for context in CodeContext
        }

    def get_improvement_suggestions(self) -> List[str]:
        """
        Get personalized improvement suggestions.

        Returns:
            List of suggestions based on patterns
        """
        suggestions = []
        weak_spots = self.get_weak_spots()

        for pattern, frequency in weak_spots:
            if pattern.preventions:
                suggestions.append(
                    f"You make {pattern.description} - try: {', '.join(pattern.preventions)}"
                )

        return suggestions

    def _get_preventions(self, issue_type: IssueType, code_context: CodeContext) -> List[str]:
        """Get prevention strategies for an issue type in context."""
        preventions_map = {
            (IssueType.LOGIC, CodeContext.LOOPS): [
                "Use range(len(array)) for cleaner indexing",
                "Consider enumerate() for safer iteration",
                "Add explicit boundary checks",
            ],
            (IssueType.SECURITY, CodeContext.VALIDATION): [
                "Always validate user input before use",
                "Use type hints for clarity",
                "Add assertion checks in tests",
            ],
            (IssueType.RELIABILITY, CodeContext.ASYNC): [
                "Always await async calls",
                "Use proper exception handling",
                "Test concurrent scenarios explicitly",
            ],
            (IssueType.PERFORMANCE, CodeContext.DATABASE): [
                "Add query limits",
                "Use connection pooling",
                "Monitor slow queries",
            ],
            (IssueType.RELIABILITY, CodeContext.ERROR_HANDLING): [
                "Catch specific exceptions",
                "Log errors with context",
                "Provide meaningful error messages",
            ],
        }

        return preventions_map.get((issue_type, code_context), [
            "Review code carefully",
            "Add comprehensive tests",
            "Use linting tools",
        ])

    def clear_patterns(self) -> None:
        """Reset all learned patterns."""
        self.patterns.clear()
        self.issue_counts = {t: 0 for t in IssueType}
        self.context_counts = {c: 0 for c in CodeContext}
