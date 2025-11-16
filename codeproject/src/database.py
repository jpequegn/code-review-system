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
    Float,
    Boolean,
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


class FeedbackType(str, Enum):
    """Type of feedback on a finding."""

    HELPFUL = "helpful"  # Finding was useful and accurate
    FALSE_POSITIVE = "false_positive"  # Not relevant or incorrect
    MISSED = "missed"  # We should have caught this
    RESOLVED = "resolved"  # Issue was fixed
    RECURRING = "recurring"  # This issue keeps happening


class IssueValidation(str, Enum):
    """Validation outcome of a finding."""

    UNVALIDATED = "unvalidated"  # No validation yet
    CONFIRMED = "confirmed"  # User confirmed the issue exists
    PARTIALLY_VALID = "partially_valid"  # Some aspects are valid
    FALSE = "false"  # Issue doesn't actually exist
    FIXED = "fixed"  # Issue was fixed


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


class FindingFeedback(Base):
    """
    Represents user feedback on a finding.

    Tracks whether findings were helpful, false positives, etc.
    This data is used to learn and adapt severity over time.
    """

    __tablename__ = "finding_feedback"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to finding
    finding_id = Column(
        Integer,
        ForeignKey("findings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Type of feedback
    feedback_type = Column(
        SQLEnum(FeedbackType), nullable=False, index=True
    )

    # Validation status of the finding
    validation = Column(
        SQLEnum(IssueValidation), default=IssueValidation.UNVALIDATED, nullable=False, index=True
    )

    # Optional notes from user
    user_notes = Column(Text, nullable=True)

    # Severity adjustment suggested by feedback (-2 to +2)
    severity_adjustment = Column(Integer, default=0, nullable=False)

    # Was this finding helpful? (boolean)
    helpful = Column(Boolean, default=None, nullable=True)

    # Timestamp
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<FindingFeedback(finding={self.finding_id}, type={self.feedback_type})>"


class ProductionIssue(Base):
    """
    Represents a bug that occurred in production.

    Links production issues back to findings that could have caught them.
    Used to measure false negative rate and learn patterns.
    """

    __tablename__ = "production_issues"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Repository URL
    repo_url = Column(String(512), nullable=False, index=True)

    # Description of the bug
    description = Column(Text, nullable=False)

    # Date the bug was discovered
    date_discovered = Column(DateTime, nullable=False, index=True)

    # Severity of the bug
    severity = Column(SQLEnum(FindingSeverity), nullable=False, index=True)

    # Time to fix in minutes (optional)
    time_to_fix_minutes = Column(Integer, nullable=True)

    # File affected (if known)
    file_path = Column(String(512), nullable=True, index=True)

    # Related findings IDs (JSON list as string)
    related_finding_ids = Column(String(512), nullable=True)

    # Timestamp when reported
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<ProductionIssue(severity={self.severity}, date={self.date_discovered.date()})>"


class LearningMetrics(Base):
    """
    Stores learning metrics tracking system accuracy over time.

    Metrics like precision, recall, false positive rate per category.
    Used to adapt analysis and learn personal thresholds.
    """

    __tablename__ = "learning_metrics"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Repository URL (for per-project metrics)
    repo_url = Column(String(512), nullable=False, index=True)

    # Category of findings (SECURITY, PERFORMANCE, BEST_PRACTICE)
    category = Column(SQLEnum(FindingCategory), nullable=False, index=True)

    # Severity level
    severity = Column(SQLEnum(FindingSeverity), nullable=False, index=True)

    # Accuracy metrics
    total_findings = Column(Integer, default=0, nullable=False)
    confirmed_findings = Column(Integer, default=0, nullable=False)
    false_positives = Column(Integer, default=0, nullable=False)
    false_negatives = Column(Integer, default=0, nullable=False)

    # Calculated metrics (as percentages: 0-100)
    accuracy = Column(Float, default=0.0, nullable=False)
    precision = Column(Float, default=0.0, nullable=False)
    recall = Column(Float, default=0.0, nullable=False)
    false_positive_rate = Column(Float, default=0.0, nullable=False)

    # Personal threshold for this category/severity
    confidence_threshold = Column(Float, default=0.5, nullable=False)

    # Last updated timestamp
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<LearningMetrics({self.category}/{self.severity}, accuracy={self.accuracy:.1f}%)>"


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
