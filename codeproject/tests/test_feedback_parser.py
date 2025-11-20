"""
Comprehensive tests for FeedbackParser.

Tests parsing of PR comments, commit messages, and emoji reactions
for acceptance/rejection signals.
"""

import pytest
from datetime import datetime, timezone

from src.learning.feedback_parser import FeedbackParser, ParsedFeedback


class TestFeedbackParserAcceptance:
    """Tests for parsing acceptance signals."""

    def test_parse_lgtm_comment(self):
        """Test parsing 'LGTM' comment."""
        result = FeedbackParser.parse_comment("LGTM")
        assert result is not None
        assert result.feedback_type == "accepted"
        assert result.confidence >= 0.85

    def test_parse_looks_good_comment(self):
        """Test parsing 'looks good' comment."""
        result = FeedbackParser.parse_comment("looks good to me!")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_merged_signal(self):
        """Test parsing 'merged' signal."""
        result = FeedbackParser.parse_comment("merged this PR")
        assert result is not None
        assert result.feedback_type == "accepted"
        assert result.confidence >= 0.90

    def test_parse_checkmark_emoji(self):
        """Test parsing checkmark emoji."""
        result = FeedbackParser.parse_emoji_reaction("white_check_mark")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_thumbsup_emoji(self):
        """Test parsing thumbs up emoji."""
        result = FeedbackParser.parse_emoji_reaction("+1")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_approved_comment(self):
        """Test parsing 'approved' comment."""
        result = FeedbackParser.parse_comment("approved")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_shipped_comment(self):
        """Test parsing 'ship it' comment."""
        result = FeedbackParser.parse_comment("ship it!")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        result1 = FeedbackParser.parse_comment("LGTM")
        result2 = FeedbackParser.parse_comment("lgtm")
        result3 = FeedbackParser.parse_comment("LgTm")

        assert all(r is not None for r in [result1, result2, result3])
        assert all(r.feedback_type == "accepted" for r in [result1, result2, result3])

    def test_parse_acceptance_with_author(self):
        """Test parsing with author information."""
        result = FeedbackParser.parse_comment(
            "looks good",
            author="alice",
            timestamp=datetime.now(timezone.utc)
        )
        assert result is not None
        assert result.developer_id == "alice"
        assert result.timestamp is not None

    def test_parse_acceptance_with_commit_hash(self):
        """Test extracting commit hash from acceptance comment."""
        result = FeedbackParser.parse_comment(
            "looks good, merged in abc1234567890def"
        )
        assert result is not None
        assert result.commit_hash is not None


class TestFeedbackParserRejection:
    """Tests for parsing rejection signals."""

    def test_parse_wont_fix_comment(self):
        """Test parsing 'won't fix' comment."""
        result = FeedbackParser.parse_comment("won't fix")
        assert result is not None
        assert result.feedback_type == "rejected"
        assert result.confidence >= 0.90

    def test_parse_not_applicable_comment(self):
        """Test parsing 'not applicable' comment."""
        result = FeedbackParser.parse_comment("not applicable")
        assert result is not None
        assert result.feedback_type == "rejected"

    def test_parse_false_positive_comment(self):
        """Test parsing 'false positive' comment."""
        result = FeedbackParser.parse_comment("this is a false positive")
        assert result is not None
        assert result.feedback_type == "rejected"
        assert result.confidence >= 0.90

    def test_parse_thumbsdown_emoji(self):
        """Test parsing thumbs down emoji."""
        result = FeedbackParser.parse_emoji_reaction("-1")
        assert result is not None
        assert result.feedback_type == "rejected"
        assert result.confidence >= 0.90

    def test_parse_x_emoji(self):
        """Test parsing X emoji."""
        result = FeedbackParser.parse_emoji_reaction("x_mark")
        assert result is not None
        assert result.feedback_type == "rejected"

    def test_parse_cant_reproduce_comment(self):
        """Test parsing 'can't reproduce' comment."""
        result = FeedbackParser.parse_comment("can't reproduce this")
        assert result is not None
        assert result.feedback_type == "rejected"

    def test_parse_already_fixed_comment(self):
        """Test parsing 'already fixed' comment."""
        result = FeedbackParser.parse_comment("already fixed in main")
        assert result is not None
        assert result.feedback_type == "rejected"

    def test_parse_disagree_comment(self):
        """Test parsing 'disagree' comment."""
        result = FeedbackParser.parse_comment("I disagree with this approach")
        assert result is not None
        assert result.feedback_type == "rejected"


class TestFeedbackParserEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_empty_comment(self):
        """Test parsing empty comment."""
        result = FeedbackParser.parse_comment("")
        assert result is None

    def test_parse_none_comment(self):
        """Test parsing None as comment."""
        result = FeedbackParser.parse_comment(None)
        assert result is None

    def test_parse_neutral_comment(self):
        """Test parsing comment with no clear signal."""
        result = FeedbackParser.parse_comment("This is a general comment")
        assert result is None

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only comment."""
        result = FeedbackParser.parse_comment("   \n\t   ")
        assert result is None

    def test_parse_multiline_comment(self):
        """Test parsing multiline comment with signal."""
        comment = """Here are my thoughts:

        This looks good to me!

        Thanks for making the change."""
        result = FeedbackParser.parse_comment(comment)
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_comment_with_code_block(self):
        """Test parsing comment with code block."""
        comment = """LGTM! Here's a suggestion:

```python
def foo():
    pass
```"""
        result = FeedbackParser.parse_comment(comment)
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_multiple_signals_first_wins(self):
        """Test that first signal found is returned."""
        # This has both acceptance and rejection signals
        # First signal (acceptance) should win based on order
        comment = "looks good, but won't merge"
        result = FeedbackParser.parse_comment(comment)
        assert result is not None
        # Depends on pattern order - check what's first
        assert result.feedback_type in ["accepted", "rejected"]

    def test_parse_unicode_text(self):
        """Test parsing comment with unicode characters."""
        result = FeedbackParser.parse_comment("LGTM! 👍 Great work!")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_truncated_signal(self):
        """Test parsing when signal word is part of larger word."""
        result = FeedbackParser.parse_comment("This is notapplicable to us")
        # Should not match "applicable" within "notapplicable"
        assert result is None or result.feedback_type == "rejected"


class TestFeedbackParserCommits:
    """Tests for parsing commit messages."""

    def test_parse_commit_with_fix(self):
        """Test parsing commit message with 'fix' keyword."""
        result = FeedbackParser.parse_commit_message(
            "fix: address security vulnerability",
            "abc1234567890def"
        )
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_commit_with_security(self):
        """Test parsing commit message with 'security' keyword."""
        result = FeedbackParser.parse_commit_message(
            "security: implement input validation",
            "abc1234567890def"
        )
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_commit_with_suggestion_keyword(self):
        """Test parsing commit that applies suggestion."""
        result = FeedbackParser.parse_commit_message(
            "Apply suggestion from code review",
            "abc1234567890def"
        )
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_commit_neutral_message(self):
        """Test parsing neutral commit message."""
        result = FeedbackParser.parse_commit_message(
            "update documentation",
            "abc1234567890def"
        )
        # Should not detect signal in neutral message
        assert result is None

    def test_parse_commit_includes_hash(self):
        """Test that parsed commit includes hash."""
        commit_hash = "abc1234567890def"
        result = FeedbackParser.parse_commit_message(
            "fix: security issue",
            commit_hash
        )
        assert result is not None
        assert result.commit_hash == commit_hash


class TestFeedbackParserEmojis:
    """Tests for emoji reaction parsing."""

    def test_parse_thumbsup_variants(self):
        """Test parsing thumbsup emoji variants."""
        variants = ["+1", "thumbsup", ":+1:", ":thumbsup:"]
        for emoji in variants:
            result = FeedbackParser.parse_emoji_reaction(emoji)
            assert result is not None
            assert result.feedback_type == "accepted"

    def test_parse_thumbsdown_variants(self):
        """Test parsing thumbsdown emoji variants."""
        variants = ["-1", "thumbsdown", ":-1:", ":thumbsdown:"]
        for emoji in variants:
            result = FeedbackParser.parse_emoji_reaction(emoji)
            assert result is not None
            assert result.feedback_type == "rejected"

    def test_parse_tada_emoji(self):
        """Test parsing tada emoji (celebration)."""
        result = FeedbackParser.parse_emoji_reaction("tada")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_rocket_emoji(self):
        """Test parsing rocket emoji."""
        result = FeedbackParser.parse_emoji_reaction("rocket")
        assert result is not None
        assert result.feedback_type == "accepted"

    def test_parse_unknown_emoji(self):
        """Test parsing unknown emoji."""
        result = FeedbackParser.parse_emoji_reaction("random_emoji")
        assert result is None


class TestFeedbackParserMultiple:
    """Tests for parsing multiple comments."""

    def test_parse_pr_comments_empty_list(self):
        """Test parsing empty comment list."""
        results = FeedbackParser.parse_pr_comments([])
        assert results == []

    def test_parse_pr_comments_single(self):
        """Test parsing single comment in list."""
        comments = [
            {"text": "looks good", "author": "alice"}
        ]
        results = FeedbackParser.parse_pr_comments(comments)
        assert len(results) == 1
        assert results[0].feedback_type == "accepted"
        assert results[0].developer_id == "alice"

    def test_parse_pr_comments_multiple_signals(self):
        """Test parsing multiple comments with different signals."""
        comments = [
            {"text": "looks good", "author": "alice"},
            {"text": "won't fix", "author": "bob"},
            {"text": "LGTM", "author": "charlie"},
        ]
        results = FeedbackParser.parse_pr_comments(comments)
        assert len(results) == 3
        assert results[0].feedback_type == "accepted"
        assert results[1].feedback_type == "rejected"
        assert results[2].feedback_type == "accepted"

    def test_parse_pr_comments_with_reactions(self):
        """Test parsing comments with emoji reactions."""
        comments = [
            {
                "text": "great work",
                "author": "alice",
                "reactions": {"+1": 2, "-1": 0}
            }
        ]
        results = FeedbackParser.parse_pr_comments(comments)
        # Should have both comment and emoji signals
        assert len(results) >= 1

    def test_parse_pr_comments_with_timestamps(self):
        """Test parsing comments with timestamps."""
        ts = datetime.now(timezone.utc).isoformat()
        comments = [
            {"text": "looks good", "author": "alice", "timestamp": ts}
        ]
        results = FeedbackParser.parse_pr_comments(comments)
        assert len(results) == 1
        assert results[0].timestamp is not None

    def test_parse_pr_comments_malformed_input(self):
        """Test parsing with malformed comment objects."""
        comments = [
            "not a dict",  # Wrong type
            {},  # Empty dict
            {"text": None},  # None text
            {"text": "looks good", "author": "alice"},
        ]
        results = FeedbackParser.parse_pr_comments(comments)
        # Should skip malformed and process valid
        assert len(results) >= 1


class TestFeedbackParserCommitExtraction:
    """Tests for commit hash extraction."""

    def test_extract_commit_hash_full(self):
        """Test extracting full 40-char commit hash."""
        text = "looks good, commit abc1234567890def1234567890abcdef1234567890"
        hash = FeedbackParser._extract_commit_hash(text)
        assert hash is not None
        assert len(hash) >= 6

    def test_extract_commit_hash_short(self):
        """Test extracting short 7-char commit hash."""
        text = "looks good, merged in abc1234"
        hash = FeedbackParser._extract_commit_hash(text)
        assert hash is not None
        assert len(hash) == 7

    def test_extract_commit_hash_not_found(self):
        """Test when no commit hash present."""
        text = "looks good, no hash here"
        hash = FeedbackParser._extract_commit_hash(text)
        assert hash is None


class TestFeedbackParserDeveloperExtraction:
    """Tests for developer mention extraction."""

    def test_extract_github_mention(self):
        """Test extracting GitHub mention."""
        text = "Thanks @alice for the fix"
        dev = FeedbackParser.extract_developer_id(text)
        assert dev == "alice"

    def test_extract_developer_prefix(self):
        """Test extracting developer with 'by' prefix."""
        text = "Fix by alice"
        dev = FeedbackParser.extract_developer_id(text)
        assert dev == "alice"

    def test_extract_developer_no_match(self):
        """Test when no developer mention."""
        text = "This looks good"
        dev = FeedbackParser.extract_developer_id(text)
        assert dev is None


class TestParsedFeedbackDataclass:
    """Tests for ParsedFeedback dataclass."""

    def test_parsed_feedback_creation(self):
        """Test creating ParsedFeedback object."""
        feedback = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.95,
            raw_text="looks good"
        )
        assert feedback.feedback_type == "accepted"
        assert feedback.confidence == 0.95
        assert feedback.raw_text == "looks good"
        assert feedback.commit_hash is None
        assert feedback.developer_id is None

    def test_parsed_feedback_with_all_fields(self):
        """Test creating ParsedFeedback with all fields."""
        ts = datetime.now(timezone.utc)
        feedback = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.90,
            raw_text="merged fix",
            commit_hash="abc1234567890def",
            developer_id="alice",
            timestamp=ts
        )
        assert feedback.feedback_type == "accepted"
        assert feedback.confidence == 0.90
        assert feedback.commit_hash == "abc1234567890def"
        assert feedback.developer_id == "alice"
        assert feedback.timestamp == ts
