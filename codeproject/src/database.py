"""
Database configuration, session management, and SQLAlchemy ORM models.

Provides:
- Database engine initialization
- Session factory
- SQLAlchemy ORM models for Review and Finding
- FastAPI dependency for database sessions
"""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from src.config import settings

# ============================================================================
# Database Engine & Session Configuration
# ============================================================================

# Create database engine using settings
engine = create_engine(
    settings.database_url,
    # SQLite-specific configuration
    connect_args=(
        {"check_same_thread": False} if "sqlite" in settings.database_url else {}
    ),
    # Connection pooling for production
    pool_size=10 if "postgresql" in settings.database_url else 0,
    max_overflow=20 if "postgresql" in settings.database_url else 0,
)

# Enable foreign key constraints for SQLite
if "sqlite" in settings.database_url:

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Session factory for creating database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()


# ============================================================================
# Enums for ORM Models
# ============================================================================


class ReviewStatus(str, Enum):
    """Status of a code review."""

    PENDING = "pending"  # Waiting to be analyzed
    ANALYZING = "analyzing"  # Currently being analyzed by LLM
    COMPLETED = "completed"  # Analysis complete
    FAILED = "failed"  # Analysis failed with error


class FindingSeverity(str, Enum):
    """Severity level of a security/performance finding."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Should be considered
    LOW = "low"  # Nice to have


class FindingCategory(str, Enum):
    """Category of a code finding."""

    SECURITY = "security"  # Security vulnerability
    PERFORMANCE = "performance"  # Performance/scalability issue
    BEST_PRACTICE = "best_practice"  # Code quality/best practice


# ============================================================================
# ORM Models
# ============================================================================


class Review(Base):
    """
    Represents a code review of a GitHub pull request.

    Tracks the review status and metadata for each PR analyzed by the system.
    """

    __tablename__ = "reviews"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Unique PR identifier (GitHub PR number)
    pr_id = Column(Integer, unique=True, index=True, nullable=False)

    # Repository URL
    repo_url = Column(String(512), nullable=False)

    # Branch being reviewed
    branch = Column(String(255), nullable=False)

    # Commit SHA being analyzed
    commit_sha = Column(String(40), nullable=False, index=True)

    # Review status (pending, analyzing, completed, failed)
    status = Column(
        SQLEnum(ReviewStatus), default=ReviewStatus.PENDING, nullable=False, index=True
    )

    # Timestamps
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at = Column(DateTime, nullable=True)

    # Relationship to findings
    findings = relationship(
        "Finding", back_populates="review", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Review(id={self.id}, pr_id={self.pr_id}, status={self.status})>"


class Finding(Base):
    """
    Represents a security or performance finding in a code review.

    Each finding is associated with a specific review and includes
    location information and suggested fixes.
    """

    __tablename__ = "findings"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to review
    review_id = Column(
        Integer,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Finding category (security, performance, best_practice)
    category = Column(SQLEnum(FindingCategory), nullable=False, index=True)

    # Severity level (critical, high, medium, low)
    severity = Column(SQLEnum(FindingSeverity), nullable=False, index=True)

    # Short title of the finding
    title = Column(String(255), nullable=False)

    # Detailed description of the issue
    description = Column(Text, nullable=False)

    # File path where the issue was found
    file_path = Column(String(512), nullable=False)

    # Line number in the file (can be null for file-level issues)
    line_number = Column(Integer, nullable=True)

    # Suggested fix or remediation
    suggested_fix = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship to review
    review = relationship("Review", back_populates="findings")

    def __repr__(self) -> str:
        return f"<Finding(id={self.id}, review_id={self.review_id}, severity={self.severity})>"


class CodeMetrics(Base):
    """
    Represents code metrics for a specific file at a point in time.

    Stores quantitative metrics like complexity, coupling, and maintainability.
    """

    __tablename__ = "code_metrics"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to review
    review_id = Column(
        Integer,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # File path being analyzed
    file_path = Column(String(512), nullable=False, index=True)

    # Programming language
    language = Column(String(50), nullable=False)

    # Complexity metrics
    cyclomatic_complexity = Column(Integer, nullable=False)
    cognitive_complexity = Column(Integer, nullable=False)
    max_nesting_depth = Column(Integer, nullable=False)

    # Size metrics
    total_lines = Column(Integer, nullable=False)
    code_lines = Column(Integer, nullable=False)
    average_function_length = Column(Integer, nullable=False)

    # Structure metrics
    function_count = Column(Integer, nullable=False)
    class_count = Column(Integer, nullable=False)
    import_count = Column(Integer, nullable=False)

    # Quality metrics
    average_complexity = Column(Integer, nullable=False)
    max_complexity = Column(Integer, nullable=False)

    # Timestamp
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<CodeMetrics(file={self.file_path}, cc={self.cyclomatic_complexity})>"


class MetricsHistory(Base):
    """
    Tracks metrics history for trend analysis.

    Records metrics snapshots at specific points in time (per commit/review).
    """

    __tablename__ = "metrics_history"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Git commit SHA
    commit_sha = Column(String(40), nullable=False, index=True, unique=True)

    # Repository URL
    repo_url = Column(String(512), nullable=False, index=True)

    # Snapshot metrics
    total_lines = Column(Integer, nullable=False)
    average_complexity = Column(Integer, nullable=False)
    max_complexity = Column(Integer, nullable=False)
    high_complexity_file_count = Column(Integer, nullable=False)

    # Timestamp
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<MetricsHistory(sha={self.commit_sha[:8]}, avg_cc={self.average_complexity})>"


class MetricsBaseline(Base):
    """
    Stores baseline metrics for a project for comparison.

    Used to identify when metrics deviate from project norms.
    """

    __tablename__ = "metrics_baseline"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Repository URL (unique per project)
    repo_url = Column(String(512), nullable=False, index=True, unique=True)

    # Baseline metrics (calculated from project history)
    average_complexity = Column(Integer, nullable=False)
    median_complexity = Column(Integer, nullable=False)
    p95_complexity = Column(Integer, nullable=False)

    # File metrics
    average_file_size = Column(Integer, nullable=False)
    average_function_length = Column(Integer, nullable=False)

    # Last updated timestamp
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<MetricsBaseline(repo={self.repo_url}, avg_cc={self.average_complexity})>"


# ============================================================================
# Database Initialization
# ============================================================================


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Should be called once on application startup or during setup.
    """
    Base.metadata.create_all(bind=engine)


# ============================================================================
# FastAPI Dependency
# ============================================================================


def get_db() -> Session:
    """
    FastAPI dependency that provides a database session.

    Yields a new database session and ensures it's closed after use.
    Can be used in FastAPI route handlers:

    Example:
        @app.get("/reviews")
        def get_reviews(db: Session = Depends(get_db)):
            return db.query(Review).all()

    Yields:
        Session: A SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
