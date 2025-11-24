"""
Cache Manager - Multi-layer caching with TTL and invalidation

Implements intelligent caching for Phase 5 performance:
- In-memory cache (10 minute TTL) for fast access
- Database cache tables for persistence across restarts
- Invalidation strategy on data changes
- Cache statistics and hit rate tracking

Cache Types:
- Metrics: Team metrics, acceptance rates, accuracy
- Paths: Learning paths, recommendations
- Trends: Vulnerability trends, historical analysis
- Anti-patterns: Detected patterns and prevalence
"""

import json
import time
from typing import Optional, Any, Dict, Tuple
from datetime import datetime, timezone, timedelta
from functools import wraps
import hashlib
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, DateTime, Float

from src.database import Base
from src.monitoring import get_metrics


class CacheType(str, Enum):
    """Types of cached data."""
    METRICS = "metrics"
    PATHS = "paths"
    TRENDS = "trends"
    PATTERNS = "patterns"
    ROI = "roi"


class CacheEntry(Base):
    """Database table for persistent cache storage."""

    __tablename__ = "cache_entries"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Cache key (repo_url + data type + params hash)
    key = Column(String(512), nullable=False, unique=True, index=True)

    # Cache type
    cache_type = Column(String(50), nullable=False, index=True)

    # Cached data (JSON)
    value = Column(Text, nullable=False)

    # Time to live (seconds)
    ttl_seconds = Column(Integer, default=600, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    hit_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime, nullable=True)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        now = datetime.now(timezone.utc)
        expires = self.expires_at.replace(tzinfo=timezone.utc) if self.expires_at.tzinfo is None else self.expires_at
        return now > expires

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)


class CacheStatistics:
    """Tracks cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.evictions = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses

    def to_dict(self) -> Dict:
        """Export statistics as dict."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "total_requests": self.total_requests,
        }


class CacheManager:
    """
    Multi-layer cache manager for Phase 5 performance.

    Provides:
    - In-memory cache with TTL for fast access
    - Database persistence for cache across restarts
    - Automatic invalidation on data changes
    - Cache statistics and hit rate tracking
    - Decorator for easy cache integration
    """

    def __init__(self, db: Session, memory_cache: Optional[Dict] = None):
        """
        Initialize cache manager.

        Args:
            db: SQLAlchemy database session
            memory_cache: Optional pre-existing memory cache dict
        """
        self.db = db
        self.memory_cache = memory_cache or {}
        self.stats = CacheStatistics()
        self.default_ttl = 600  # 10 minutes

    # ============================================================================
    # Cache Operations
    # ============================================================================

    def _make_key(self, cache_type: CacheType, repo_url: str, params: Optional[Dict] = None) -> str:
        """
        Create a cache key from type, repo, and params.

        Args:
            cache_type: Type of cache entry
            repo_url: Repository URL
            params: Optional parameters to hash

        Returns:
            Cache key string
        """
        key_parts = f"{cache_type.value}:{repo_url}"
        if params:
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            key_parts += f":{params_hash}"
        return key_parts

    def get(self, cache_type: CacheType, repo_url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Get value from cache (memory first, then database).

        Args:
            cache_type: Type of cache entry
            repo_url: Repository URL
            params: Optional parameters

        Returns:
            Cached value or None if not found/expired
        """
        key = self._make_key(cache_type, repo_url, params)
        metrics = get_metrics()

        # Try memory cache first
        if key in self.memory_cache:
            entry_data = self.memory_cache[key]
            if entry_data["expires_at"] > datetime.now(timezone.utc):
                self.stats.hits += 1
                metrics.register_counter("cache_hits_total").increment()
                return entry_data["value"]
            else:
                # Expired in memory, remove it
                del self.memory_cache[key]

        # Try database cache
        cache_entry = (
            self.db.query(CacheEntry)
            .filter(CacheEntry.key == key)
            .first()
        )

        if cache_entry and not cache_entry.is_expired():
            cache_entry.record_hit()
            self.db.commit()

            # Store in memory for next access
            value = json.loads(cache_entry.value)
            self.memory_cache[key] = {
                "value": value,
                "expires_at": cache_entry.expires_at,
            }

            self.stats.hits += 1
            metrics.register_counter("cache_hits_total").increment()
            return value

        # Cache miss
        self.stats.misses += 1
        metrics.register_counter("cache_misses_total").increment()
        return None

    def set(
        self,
        cache_type: CacheType,
        repo_url: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:
        """
        Set value in cache (both memory and database).

        Args:
            cache_type: Type of cache entry
            repo_url: Repository URL
            value: Value to cache
            ttl_seconds: Time to live in seconds (default: 600)
            params: Optional parameters
        """
        key = self._make_key(cache_type, repo_url, params)
        ttl = ttl_seconds or self.default_ttl
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl)

        # Store in memory
        self.memory_cache[key] = {
            "value": value,
            "expires_at": expires_at,
        }

        # Store in database
        cache_entry = (
            self.db.query(CacheEntry)
            .filter(CacheEntry.key == key)
            .first()
        )

        if cache_entry:
            # Update existing
            cache_entry.value = json.dumps(value)
            cache_entry.ttl_seconds = ttl
            cache_entry.expires_at = expires_at
            cache_entry.created_at = now
            cache_entry.hit_count = 0
        else:
            # Create new
            cache_entry = CacheEntry(
                key=key,
                cache_type=cache_type.value,
                value=json.dumps(value),
                ttl_seconds=ttl,
                expires_at=expires_at,
            )
            self.db.add(cache_entry)

        self.db.commit()

    def invalidate(
        self,
        cache_type: Optional[CacheType] = None,
        repo_url: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        Args:
            cache_type: Optional type to filter (None = all types)
            repo_url: Optional repo URL to filter (None = all repos)

        Returns:
            Count of invalidated entries
        """
        count = 0
        metrics = get_metrics()

        # Invalidate memory cache
        keys_to_delete = []
        for key in self.memory_cache:
            if cache_type and not key.startswith(cache_type.value):
                continue
            if repo_url and repo_url not in key:
                continue
            keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.memory_cache[key]
            count += 1

        # Invalidate database cache
        query = self.db.query(CacheEntry)
        if cache_type:
            query = query.filter(CacheEntry.cache_type == cache_type.value)
        if repo_url:
            query = query.filter(CacheEntry.key.contains(repo_url))

        count += query.delete()
        self.db.commit()

        self.stats.invalidations += count
        metrics.register_counter("cache_invalidations_total").increment(count)
        return count

    def clear_expired(self) -> int:
        """
        Clear expired cache entries from database.

        Returns:
            Count of cleared entries
        """
        now = datetime.now(timezone.utc)
        count = (
            self.db.query(CacheEntry)
            .filter(CacheEntry.expires_at < now)
            .delete()
        )
        self.db.commit()

        self.stats.evictions += count
        return count

    # ============================================================================
    # Statistics
    # ============================================================================

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        return self.stats.to_dict()

    def get_cache_info(self) -> Dict:
        """Get detailed cache information."""
        memory_entries = len(self.memory_cache)
        db_entries = self.db.query(CacheEntry).count()
        expired_entries = (
            self.db.query(CacheEntry)
            .filter(CacheEntry.expires_at < datetime.now(timezone.utc))
            .count()
        )

        metrics = get_metrics()
        metrics.register_gauge("cache_memory_entries").set(memory_entries)
        metrics.register_gauge("cache_database_entries").set(db_entries)

        hit_rate = self.stats.hit_rate
        metrics.register_gauge("insights_cache_hit_rate").set(hit_rate)

        return {
            "memory_entries": memory_entries,
            "database_entries": db_entries,
            "expired_entries": expired_entries,
            "statistics": self.get_statistics(),
        }

    # ============================================================================
    # Decorator for Function Caching
    # ============================================================================

    def cached(
        self,
        cache_type: CacheType,
        ttl_seconds: int = 600,
        key_params: Optional[Tuple[str, ...]] = None,
    ):
        """
        Decorator for caching function results.

        Args:
            cache_type: Type of cache entry
            ttl_seconds: Time to live in seconds
            key_params: Tuple of parameter names to include in cache key

        Example:
            @cache_manager.cached(CacheType.METRICS, ttl_seconds=600, key_params=('repo_url',))
            def calculate_metrics(repo_url: str) -> dict:
                return {...}
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract repo_url from kwargs or first arg
                repo_url = kwargs.get("repo_url")
                if not repo_url and len(args) > 0:
                    # Try to find repo_url in kwargs or use a default
                    repo_url = "default"

                # Build cache params from key_params
                cache_params = None
                if key_params:
                    cache_params = {}
                    for param_name in key_params:
                        if param_name in kwargs:
                            cache_params[param_name] = kwargs[param_name]

                # Try cache first
                cached_value = self.get(cache_type, repo_url, cache_params)
                if cached_value is not None:
                    return cached_value

                # Call function
                result = func(*args, **kwargs)

                # Store in cache
                self.set(cache_type, repo_url, result, ttl_seconds, cache_params)

                return result

            return wrapper
        return decorator
