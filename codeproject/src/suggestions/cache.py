"""
Suggestion Caching Module

Provides SQLite-based caching for AI-generated suggestions to avoid redundant
LLM calls for identical findings. Implements cache key generation based on
finding title, file path, and code snippet hash.

Architecture:
- Cache key: SHA256(title + file_path + code_snippet)
- TTL: Configurable (default 7 days)
- Storage: SQLite (no external dependencies)
- Hit tracking: For performance monitoring
"""

import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Key Generation
# ============================================================================


def generate_cache_key(
    title: str,
    file_path: str,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Generate cache key from finding attributes.

    Combines finding title, file path, and code snippet into a SHA256 hash.
    Used to identify cached suggestions for identical findings.

    Args:
        title: Finding title
        file_path: File path where finding occurred
        code_snippet: Optional code snippet (increases specificity)

    Returns:
        SHA256 hash as cache key (hex string)
    """
    # Combine attributes to create unique key
    key_material = f"{title}:{file_path}:{code_snippet or ''}"

    # Generate SHA256 hash
    return hashlib.sha256(key_material.encode()).hexdigest()


# ============================================================================
# Suggestion Cache
# ============================================================================


class SuggestionCache:
    """
    SQLite-based cache for AI-generated suggestions.

    Caches complete suggestion sets to avoid regenerating for identical findings.
    Implements TTL, cache invalidation, and hit/miss tracking.

    Attributes:
        db_path: Path to SQLite cache database
        ttl_days: Time-to-live for cached suggestions (default: 7)
        enabled: Whether caching is enabled
    """

    def __init__(self, db_path: Optional[str] = None, ttl_days: Optional[int] = None):
        """
        Initialize suggestion cache.

        Args:
            db_path: Path to SQLite cache database (uses same as main DB if not provided)
            ttl_days: Cache TTL in days (uses config if not provided)
        """
        self.enabled = settings.cache_suggestions

        if not self.enabled:
            logger.debug("Suggestion caching is disabled")
            return

        # Set database path
        if db_path is None:
            # Use suggestions_cache.db in same directory as main database
            db_path = str(Path(settings.database_url.replace("sqlite:///./", "")).parent / "suggestions_cache.db")

        self.db_path = db_path
        self.ttl_days = ttl_days or settings.suggestion_cache_ttl_days

        # Initialize database
        self._init_db()

        logger.info(
            f"Suggestion cache initialized: {self.db_path} (TTL: {self.ttl_days} days)"
        )

    def _init_db(self) -> None:
        """Initialize cache database and tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create suggestions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS suggestion_cache (
                        id INTEGER PRIMARY KEY,
                        cache_key TEXT NOT NULL UNIQUE,
                        title TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        auto_fix TEXT,
                        auto_fix_confidence REAL DEFAULT 0.0,
                        explanation TEXT,
                        improvement_suggestions TEXT,
                        created_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        hit_count INTEGER DEFAULT 0
                    )
                    """
                )

                # Create index on cache_key for faster lookups
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_key ON suggestion_cache(cache_key)"
                )

                # Create index on expires_at for cleanup queries
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expires_at ON suggestion_cache(expires_at)"
                )

                conn.commit()
                logger.debug(f"Cache database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
            raise

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached suggestion by key.

        Args:
            cache_key: Cache key generated from finding attributes

        Returns:
            Dictionary with cached suggestions or None if not found/expired
        """
        if not self.enabled:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Query for valid (non-expired) cache entry
                cursor.execute(
                    """
                    SELECT id, auto_fix, auto_fix_confidence, explanation, improvement_suggestions, hit_count
                    FROM suggestion_cache
                    WHERE cache_key = ? AND expires_at > datetime('now')
                    """,
                    (cache_key,),
                )

                row = cursor.fetchone()
                if not row:
                    logger.debug(f"Cache miss: {cache_key}")
                    return None

                # Update hit count
                hit_count = row["hit_count"] + 1
                cursor.execute(
                    "UPDATE suggestion_cache SET hit_count = ? WHERE id = ?",
                    (hit_count, row["id"]),
                )
                conn.commit()

                logger.debug(f"Cache hit: {cache_key} (hit #{hit_count})")

                return {
                    "auto_fix": row["auto_fix"],
                    "auto_fix_confidence": row["auto_fix_confidence"],
                    "explanation": row["explanation"],
                    "improvement_suggestions": row["improvement_suggestions"],
                    "hit_count": hit_count,
                    "cached": True,  # Mark as from cache
                }

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None

    def set(
        self,
        cache_key: str,
        title: str,
        file_path: str,
        auto_fix: Optional[str] = None,
        auto_fix_confidence: float = 0.0,
        explanation: Optional[str] = None,
        improvement_suggestions: Optional[str] = None,
    ) -> bool:
        """
        Store suggestion in cache.

        Args:
            cache_key: Cache key generated from finding attributes
            title: Finding title
            file_path: File path where finding occurred
            auto_fix: Generated code fix
            auto_fix_confidence: Confidence score for auto_fix (0.0-1.0)
            explanation: Educational explanation
            improvement_suggestions: Best practice suggestions

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=self.ttl_days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO suggestion_cache
                    (cache_key, title, file_path, auto_fix, auto_fix_confidence, explanation,
                     improvement_suggestions, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        auto_fix = excluded.auto_fix,
                        auto_fix_confidence = excluded.auto_fix_confidence,
                        explanation = excluded.explanation,
                        improvement_suggestions = excluded.improvement_suggestions,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        hit_count = 0
                    """,
                    (
                        cache_key,
                        title,
                        file_path,
                        auto_fix,
                        auto_fix_confidence,
                        explanation,
                        improvement_suggestions,
                        now,
                        expires_at,
                    ),
                )

                conn.commit()
                logger.debug(f"Cached suggestion: {cache_key}")

                return True

        except Exception as e:
            logger.error(f"Cache write failed: {e}")
            return False

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM suggestion_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()

                logger.debug(f"Invalidated cache entry: {cache_key}")
                return True

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return False

    def clear_expired(self) -> int:
        """
        Remove all expired cache entries.

        Returns:
            Number of entries deleted
        """
        if not self.enabled:
            return 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM suggestion_cache WHERE expires_at <= datetime('now')"
                )
                conn.commit()

                deleted = cursor.rowcount
                logger.debug(f"Cleared {deleted} expired cache entries")

                return deleted

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear entire cache.

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM suggestion_cache")
                conn.commit()

                logger.info("Cleared entire suggestion cache")
                return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size, hit count, TTL, etc.
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total entries
                cursor.execute("SELECT COUNT(*) FROM suggestion_cache")
                total_entries = cursor.fetchone()[0]

                # Get non-expired entries
                cursor.execute(
                    "SELECT COUNT(*) FROM suggestion_cache WHERE expires_at > datetime('now')"
                )
                valid_entries = cursor.fetchone()[0]

                # Get total hits
                cursor.execute("SELECT SUM(hit_count) FROM suggestion_cache")
                total_hits = cursor.fetchone()[0] or 0

                # Get size
                cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                size_bytes = cursor.fetchone()[0] or 0

                return {
                    "enabled": True,
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "expired_entries": total_entries - valid_entries,
                    "total_hits": total_hits,
                    "ttl_days": self.ttl_days,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "db_path": self.db_path,
                }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}


# Global cache instance
_cache_instance: Optional[SuggestionCache] = None


def get_cache() -> SuggestionCache:
    """
    Get global suggestion cache instance (lazy initialization).

    Returns:
        Global SuggestionCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = SuggestionCache()

    return _cache_instance
