"""
Migration: Add AI suggestion fields to Finding model

This migration adds three new optional columns to the findings table
to persist AI-generated suggestions alongside findings:
- auto_fix: Auto-generated code patches (for high/critical findings)
- explanation: Educational explanation of findings
- improvement_suggestions: Best practice suggestions

Migration ID: 001
Type: Schema migration for SQLite
Status: Can be executed directly or integrated with Alembic
"""

import sqlite3
from pathlib import Path


def execute_sqlite_migration(db_path: str):
    """
    Execute migration against SQLite database.

    Adds three new TEXT columns to findings table for AI-generated suggestions.
    All columns are nullable for backward compatibility.

    Args:
        db_path: Path to SQLite database file

    Raises:
        sqlite3.OperationalError: If database operation fails (except duplicate columns)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        print(f"Applying migration: Add suggestion fields to findings table")
        print(f"  Database: {db_path}")

        # Add auto_fix column
        try:
            cursor.execute("ALTER TABLE findings ADD COLUMN auto_fix TEXT NULL")
            print("  ✓ Added auto_fix column")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠ auto_fix column already exists")
            else:
                raise

        # Add explanation column
        try:
            cursor.execute("ALTER TABLE findings ADD COLUMN explanation TEXT NULL")
            print("  ✓ Added explanation column")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠ explanation column already exists")
            else:
                raise

        # Add improvement_suggestions column
        try:
            cursor.execute(
                "ALTER TABLE findings ADD COLUMN improvement_suggestions TEXT NULL"
            )
            print("  ✓ Added improvement_suggestions column")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠ improvement_suggestions column already exists")
            else:
                raise

        conn.commit()
        print("\n✓ Migration successful: Suggestion fields added to findings table")
        return True

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


def execute_postgresql_migration(connection_string: str):
    """
    Execute migration against PostgreSQL database.

    Args:
        connection_string: PostgreSQL connection string
    """
    import psycopg2

    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()

    try:
        print(f"Applying migration: Add suggestion fields to findings table")

        statements = [
            "ALTER TABLE findings ADD COLUMN auto_fix TEXT NULL",
            "ALTER TABLE findings ADD COLUMN explanation TEXT NULL",
            "ALTER TABLE findings ADD COLUMN improvement_suggestions TEXT NULL",
        ]

        for stmt in statements:
            try:
                cursor.execute(stmt)
                print(f"  ✓ {stmt}")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    print(f"  ⚠ Column already exists")
                else:
                    raise

        conn.commit()
        print("\n✓ Migration successful")
        return True

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def downgrade_sqlite(db_path: str):
    """
    Revert migration on SQLite database.

    Removes the suggestion columns from findings table.

    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        print(f"Reverting migration: Remove suggestion fields from findings table")

        # SQLite doesn't support DROP COLUMN directly in older versions
        # We would need to recreate the table
        print("  ⚠ SQLite doesn't support DROP COLUMN directly")
        print("  To downgrade: Restore from backup or recreate table manually")
        return False

    finally:
        conn.close()


def get_migration_info() -> dict:
    """Get information about this migration."""
    return {
        "id": "001",
        "name": "Add AI suggestion fields to Finding model",
        "description": "Adds auto_fix, explanation, and improvement_suggestions columns",
        "created_at": "2025-11-19",
        "reversible": False,  # SQLite doesn't support easy column removal
        "database_compatibility": ["sqlite3", "postgresql", "mysql"],
    }


if __name__ == "__main__":
    # Allow running migration directly from command line
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default to codeproject.db in parent directory
        db_path = Path(__file__).parent.parent / "codeproject.db"

    print(f"Running migration 001: Add suggestion fields\n")
    execute_sqlite_migration(str(db_path))
