"""
Tests for suggestion caching module.

Tests cache key generation, cache operations, TTL, hit tracking,
and cache invalidation.
"""

import pytest
import time
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from src.suggestions.cache import SuggestionCache, generate_cache_key
from src.config import Settings


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cache_db():
    """Create temporary cache database for testing."""
    with TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_cache.db")
        yield db_path


@pytest.fixture
def cache(temp_cache_db):
    """Create cache instance with temp database."""
    return SuggestionCache(db_path=temp_cache_db, ttl_days=7)


# ============================================================================
# Cache Key Generation Tests
# ============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_cache_key_basic(self):
        """Cache key should be consistent for same inputs."""
        key1 = generate_cache_key("SQL Injection", "src/db.py")
        key2 = generate_cache_key("SQL Injection", "src/db.py")

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex is 64 chars

    def test_generate_cache_key_different_titles(self):
        """Different titles should produce different keys."""
        key1 = generate_cache_key("SQL Injection", "src/db.py")
        key2 = generate_cache_key("XSS Vulnerability", "src/db.py")

        assert key1 != key2

    def test_generate_cache_key_different_files(self):
        """Different files should produce different keys."""
        key1 = generate_cache_key("SQL Injection", "src/db.py")
        key2 = generate_cache_key("SQL Injection", "src/api.py")

        assert key1 != key2

    def test_generate_cache_key_with_code_snippet(self):
        """Code snippet should be included in key generation."""
        snippet1 = "user = db.query(User).filter('id = ' + user_id)"
        snippet2 = "user = db.query(User).filter(User.id == user_id)"

        key1 = generate_cache_key("SQL Injection", "src/db.py", snippet1)
        key2 = generate_cache_key("SQL Injection", "src/db.py", snippet2)

        assert key1 != key2

    def test_generate_cache_key_without_code_snippet(self):
        """Code snippet should be optional."""
        key1 = generate_cache_key("SQL Injection", "src/db.py", None)
        key2 = generate_cache_key("SQL Injection", "src/db.py", "")

        # Both should be valid (though different)
        assert len(key1) == 64
        assert len(key2) == 64


# ============================================================================
# Cache Operations Tests
# ============================================================================


class TestCacheOperations:
    """Tests for cache get/set/invalidate operations."""

    def test_cache_set_and_get(self, cache):
        """Cache should store and retrieve suggestions."""
        cache_key = generate_cache_key("SQL Injection", "src/db.py")

        # Initially not in cache
        assert cache.get(cache_key) is None

        # Store in cache
        success = cache.set(
            cache_key,
            "SQL Injection",
            "src/db.py",
            auto_fix="use parameterized queries",
            auto_fix_confidence=0.95,
            explanation="SQL injection is dangerous",
            improvement_suggestions="- Use ORM\n- Validate input",
        )

        assert success is True

        # Retrieve from cache
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached["auto_fix"] == "use parameterized queries"
        assert cached["auto_fix_confidence"] == 0.95
        assert cached["explanation"] == "SQL injection is dangerous"
        assert cached["improvement_suggestions"] == "- Use ORM\n- Validate input"

    def test_cache_get_nonexistent_key(self, cache):
        """Getting non-existent key should return None."""
        key = generate_cache_key("Nonexistent", "nowhere.py")
        assert cache.get(key) is None

    def test_cache_miss_logging(self, cache, caplog):
        """Cache misses should be logged."""
        import logging
        caplog.set_level(logging.DEBUG, logger="src.suggestions.cache")

        key = generate_cache_key("Missing", "src/test.py")
        cache.get(key)

        assert any("Cache miss" in record.message for record in caplog.records)

    def test_cache_hit_logging(self, cache, caplog):
        """Cache hits should be logged."""
        import logging
        caplog.set_level(logging.DEBUG, logger="src.suggestions.cache")

        key = generate_cache_key("Found", "src/test.py")

        cache.set(key, "Found", "src/test.py", explanation="test")
        caplog.clear()
        caplog.set_level(logging.DEBUG, logger="src.suggestions.cache")

        cache.get(key)

        assert any("Cache hit" in record.message for record in caplog.records)

    def test_cache_hit_count_increments(self, cache):
        """Hit count should increment on each cache hit."""
        key = generate_cache_key("Test", "src/test.py")

        cache.set(key, "Test", "src/test.py", explanation="test")

        # First hit
        cached1 = cache.get(key)
        assert cached1["hit_count"] == 1

        # Second hit
        cached2 = cache.get(key)
        assert cached2["hit_count"] == 2

    def test_cache_partial_suggestions(self, cache):
        """Cache should support partial suggestions (some fields None)."""
        key = generate_cache_key("Partial", "src/test.py")

        cache.set(
            key,
            "Partial",
            "src/test.py",
            explanation="only explanation",
            # auto_fix and improvements are None
        )

        cached = cache.get(key)
        assert cached["explanation"] == "only explanation"
        assert cached["auto_fix"] is None
        assert cached["improvement_suggestions"] is None

    def test_cache_invalidate(self, cache):
        """Invalidation should remove cache entry."""
        key = generate_cache_key("Invalidate", "src/test.py")

        cache.set(key, "Invalidate", "src/test.py", explanation="test")
        assert cache.get(key) is not None

        # Invalidate
        success = cache.invalidate(key)
        assert success is True

        # Should be gone now
        assert cache.get(key) is None

    def test_cache_update_overwrites(self, cache):
        """Setting same key again should overwrite."""
        key = generate_cache_key("Update", "src/test.py")

        cache.set(key, "Update", "src/test.py", explanation="old")
        cached1 = cache.get(key)
        assert cached1["explanation"] == "old"
        assert cached1["hit_count"] == 1

        # Update with new value
        cache.set(key, "Update", "src/test.py", explanation="new")
        cached2 = cache.get(key)
        assert cached2["explanation"] == "new"
        # After update and then get, hit_count should be 1 (reset on update + 1 for this get)
        assert cached2["hit_count"] == 1

    def test_cache_clear_all(self, cache):
        """Clear all should remove all entries."""
        key1 = generate_cache_key("Entry1", "src/test1.py")
        key2 = generate_cache_key("Entry2", "src/test2.py")

        cache.set(key1, "Entry1", "src/test1.py", explanation="one")
        cache.set(key2, "Entry2", "src/test2.py", explanation="two")

        assert cache.get(key1) is not None
        assert cache.get(key2) is not None

        # Clear all
        success = cache.clear_all()
        assert success is True

        assert cache.get(key1) is None
        assert cache.get(key2) is None


# ============================================================================
# TTL and Expiration Tests
# ============================================================================


class TestCacheTTLandExpiration:
    """Tests for cache TTL and expiration."""

    def test_cache_ttl_respected(self, cache):
        """Expired entries should not be retrieved."""
        key = generate_cache_key("Expire", "src/test.py")
        cache.set(key, "Expire", "src/test.py", explanation="test")

        # Verify it's in cache
        assert cache.get(key) is not None

        # Manually mark as expired in database
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE suggestion_cache SET expires_at = datetime('now', '-1 day')"
            )
            conn.commit()

        # Should be expired now
        assert cache.get(key) is None

    def test_cache_clear_expired(self, cache):
        """clear_expired should remove expired entries."""
        key = generate_cache_key("Expired", "src/test.py")
        cache.set(key, "Expired", "src/test.py", explanation="test")

        # Manually mark as expired in database
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE suggestion_cache SET expires_at = datetime('now', '-1 day')"
            )
            conn.commit()

        # Should be expired
        assert cache.get(key) is None

        # Clear expired
        deleted = cache.clear_expired()
        assert deleted >= 1

        # Should still be gone
        assert cache.get(key) is None


# ============================================================================
# Cache Statistics Tests
# ============================================================================


class TestCacheStatistics:
    """Tests for cache statistics and monitoring."""

    def test_get_stats_empty_cache(self, cache):
        """Stats should show empty cache."""
        stats = cache.get_stats()

        assert stats["enabled"] is True
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["total_hits"] == 0

    def test_get_stats_with_entries(self, cache):
        """Stats should accurately count entries and hits."""
        key1 = generate_cache_key("Entry1", "src/test1.py")
        key2 = generate_cache_key("Entry2", "src/test2.py")

        cache.set(key1, "Entry1", "src/test1.py", explanation="one")
        cache.set(key2, "Entry2", "src/test2.py", explanation="two")

        # Generate some hits
        cache.get(key1)  # 1 hit
        cache.get(key1)  # 2 hits
        cache.get(key2)  # 1 hit

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["total_hits"] == 3  # 2 + 1

    def test_get_stats_includes_metadata(self, cache):
        """Stats should include cache metadata."""
        stats = cache.get_stats()

        assert "enabled" in stats
        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert "total_hits" in stats
        assert "ttl_days" in stats
        assert "size_bytes" in stats
        assert "size_mb" in stats
        assert "db_path" in stats


# ============================================================================
# Cache Disabled Tests
# ============================================================================


class TestCacheDisabled:
    """Tests for cache when disabled."""

    def test_cache_disabled_get_returns_none(self):
        """Getting from disabled cache should return None."""
        with temporary_settings(cache_suggestions=False):
            cache = SuggestionCache()
            key = generate_cache_key("Test", "src/test.py")

            assert cache.get(key) is None

    def test_cache_disabled_set_returns_false(self):
        """Setting in disabled cache should return False."""
        with temporary_settings(cache_suggestions=False):
            cache = SuggestionCache()
            key = generate_cache_key("Test", "src/test.py")

            success = cache.set(key, "Test", "src/test.py", explanation="test")
            assert success is False

    def test_cache_disabled_stats(self):
        """Stats should show cache disabled."""
        with temporary_settings(cache_suggestions=False):
            cache = SuggestionCache()
            stats = cache.get_stats()

            assert stats["enabled"] is False


# ============================================================================
# Helper Functions
# ============================================================================


class temporary_settings:
    """Context manager to temporarily change settings."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.original_values = {}

    def __enter__(self):
        from src import config

        for key, value in self.kwargs.items():
            self.original_values[key] = getattr(config.settings, key)
            setattr(config.settings, key, value)

    def __exit__(self, *args):
        from src import config

        for key, original_value in self.original_values.items():
            setattr(config.settings, key, original_value)
