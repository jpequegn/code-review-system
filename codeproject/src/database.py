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
