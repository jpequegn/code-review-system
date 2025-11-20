"""
Feedback Parser - Extracts acceptance/rejection signals from GitHub PR comments.

Parses PR comments, commit messages, and reactions to detect whether developers
have accepted, rejected, or ignored AI-generated suggestions.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone


@dataclass
class ParsedFeedback:
    """Represents a parsed feedback signal from a developer."""

    feedback_type: str  # "accepted", "rejected", "ignored"
    confidence: float  # 0.0-1.0, confidence in the parsing
    raw_text: str  # The original text that was parsed
    commit_hash: Optional[str] = None  # Git commit if applied
    developer_id: Optional[str] = None  # GitHub username
    timestamp: Optional[datetime] = None  # When feedback was given


class FeedbackParser:
    """
    Parses GitHub PR comments, reactions, and commit messages for feedback signals.

    Detects acceptance patterns like "LGTM", "looks good", merged commits,
    rejection patterns like "won't fix", "not applicable", and ignores other comments.
    """

    # Acceptance signal patterns (ordered by confidence, descending)
    ACCEPTANCE_PATTERNS = [
        (r'\bmerged\b', 0.95),  # High confidence: explicitly merged
        (r'applied["\']?\s*(?:the\s+)?(?:fix|suggestion)', 0.90),
        (r'(?:looks\s+)?good["\']?\s*(?:to\s+me)?', 0.85),  # LGTM variants
        (r'\blgtm\b', 0.85),
        (r'\bLGTM\b', 0.85),
        (r'✓|✅', 0.90),  # Check mark emoji
        (r'thanks[,.]?\s+(?:applied|fixed|updated)', 0.80),
        (r'fixed\s+(?:in\s+)?commit', 0.85),
        (r'commit.*(?:address|fix|implement)', 0.80),
        (r'approved|approved', 0.85),  # PR approved
        (r'ship\s+it', 0.85),
        (r'ready\s+(?:to\s+)?(?:ship|merge)', 0.80),
    ]

    # Rejection signal patterns (ordered by confidence, descending)
    REJECTION_PATTERNS = [
        (r"won't\s+fix", 0.95),
        (r'not\s+(?:applicable|relevant|needed)', 0.90),
        (r'(?:already|already)\s+(?:fixed|handled|done)', 0.85),
        (r'(?:disagree|disagree)|not\s+agree', 0.85),
        (r'false\s+positive', 0.95),
        (r'(?:is\s+)?working\s+as\s+intended', 0.80),
        (r'(?:not\s+)?(?:a\s+)?(?:real\s+)?(?:issue|problem)', 0.80),
        (r'❌', 0.90),  # X mark emoji
        (r"(?:don't|do\s+not)\s+(?:think|agree)", 0.75),
        (r"can't\s+(?:reproduce|replicate)", 0.85),
        (r"won't\s+(?:do|apply)\s+this", 0.90),
    ]

    # Patterns to extract commit hashes
    COMMIT_HASH_PATTERNS = [
        r'(?:commit|fixes?|closes?|in)\s+([0-9a-f]{6,40})',
        r'\b([0-9a-f]{40})\b',
        r'\b([0-9a-f]{7,40})\b',
    ]

    # Patterns to extract developer mentions
    DEVELOPER_MENTION_PATTERNS = [
        r'@([\w-]+)',  # GitHub mention
        r'(?:by|from|author:|owner:)\s+@?([\w-]+)',
    ]

    @staticmethod
    def parse_comment(
        comment_text: str,
        author: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ParsedFeedback]:
        """
        Parse a single PR comment for feedback signals.

        Args:
            comment_text: The comment text to parse
            author: GitHub username of the comment author
            timestamp: When the comment was created

        Returns:
            ParsedFeedback if a signal is found, None otherwise
        """
        if not comment_text or not isinstance(comment_text, str):
            return None

        # Normalize text for matching (lowercase, extra whitespace)
        normalized = comment_text.lower().strip()

        # Check acceptance patterns
        for pattern, confidence in FeedbackParser.ACCEPTANCE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                commit_hash = FeedbackParser._extract_commit_hash(comment_text)
                return ParsedFeedback(
                    feedback_type="accepted",
                    confidence=confidence,
                    raw_text=comment_text,
                    commit_hash=commit_hash,
                    developer_id=author,
                    timestamp=timestamp,
                )

        # Check rejection patterns
        for pattern, confidence in FeedbackParser.REJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return ParsedFeedback(
                    feedback_type="rejected",
                    confidence=confidence,
                    raw_text=comment_text,
                    developer_id=author,
                    timestamp=timestamp,
                )

        # No clear signal found
        return None

    @staticmethod
    def parse_commit_message(
        commit_message: str,
        commit_hash: str,
        author: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ParsedFeedback]:
        """
        Parse a commit message to detect if it applied a suggestion.

        Args:
            commit_message: The commit message text
            commit_hash: The commit SHA
            author: Git author
            timestamp: Commit timestamp

        Returns:
            ParsedFeedback if signal found, None otherwise
        """
        if not commit_message:
            return None

        normalized = commit_message.lower().strip()

        # Check if commit message indicates applying a fix/suggestion
        apply_patterns = [
            r'(?:apply|implement|fix)\s+(?:suggestion|fix)',
            r'(?:address|fix)\s+.*\bsecurity\b',
            r'fix:\s+',  # Conventional commits
            r'security:\s+',
        ]

        for pattern in apply_patterns:
            if re.search(pattern, normalized):
                return ParsedFeedback(
                    feedback_type="accepted",
                    confidence=0.75,
                    raw_text=commit_message,
                    commit_hash=commit_hash,
                    developer_id=author,
                    timestamp=timestamp,
                )

        return None

    @staticmethod
    def parse_emoji_reaction(
        emoji: str, author: Optional[str] = None, timestamp: Optional[datetime] = None
    ) -> Optional[ParsedFeedback]:
        """
        Parse GitHub emoji reactions for feedback signals.

        Args:
            emoji: The emoji reaction (e.g., "+1", "-1", "tada")
            author: GitHub username who reacted
            timestamp: When reaction was added

        Returns:
            ParsedFeedback if signal found, None otherwise
        """
        acceptance_emojis = {
            "+1": 0.90,  # Thumbs up
            "thumbsup": 0.90,
            "tada": 0.80,  # Party popper
            "rocket": 0.80,  # Rocket
            "checkmark": 0.90,
            "white_check_mark": 0.90,
            "ok": 0.85,
        }

        rejection_emojis = {
            "-1": 0.90,  # Thumbs down
            "thumbsdown": 0.90,
            "x": 0.90,
            "x_mark": 0.90,
            "thinking_face": 0.70,  # More ambiguous
            "confused": 0.70,
        }

        normalized_emoji = emoji.lower().replace(":", "").strip()

        if normalized_emoji in acceptance_emojis:
            return ParsedFeedback(
                feedback_type="accepted",
                confidence=acceptance_emojis[normalized_emoji],
                raw_text=f":{emoji}:",
                developer_id=author,
                timestamp=timestamp,
            )

        if normalized_emoji in rejection_emojis:
            return ParsedFeedback(
                feedback_type="rejected",
                confidence=rejection_emojis[normalized_emoji],
                raw_text=f":{emoji}:",
                developer_id=author,
                timestamp=timestamp,
            )

        return None

    @staticmethod
    def parse_pr_comments(
        comments: List[Dict],
    ) -> List[ParsedFeedback]:
        """
        Parse multiple PR comments for feedback signals.

        Args:
            comments: List of comment dicts with keys: text, author, timestamp (optional)

        Returns:
            List of ParsedFeedback objects
        """
        feedbacks = []

        for comment in comments:
            if not isinstance(comment, dict):
                continue

            text = comment.get("text") or comment.get("body") or ""
            author = comment.get("author") or comment.get("user")
            timestamp_str = comment.get("timestamp")

            # Parse timestamp if provided
            timestamp = None
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        # Try ISO format first
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        timestamp = timestamp_str
                except (ValueError, AttributeError):
                    pass

            # Parse comment text
            parsed = FeedbackParser.parse_comment(text, author, timestamp)
            if parsed:
                feedbacks.append(parsed)

            # Also check for emoji reactions
            reactions = comment.get("reactions", {})
            for emoji, count in reactions.items():
                if count > 0:
                    emoji_feedback = FeedbackParser.parse_emoji_reaction(emoji, author)
                    if emoji_feedback:
                        feedbacks.append(emoji_feedback)

        return feedbacks

    @staticmethod
    def _extract_commit_hash(text: str) -> Optional[str]:
        """Extract commit hash from text."""
        for pattern in FeedbackParser.COMMIT_HASH_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def extract_developer_id(text: str) -> Optional[str]:
        """Extract developer mention from text."""
        for pattern in FeedbackParser.DEVELOPER_MENTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
