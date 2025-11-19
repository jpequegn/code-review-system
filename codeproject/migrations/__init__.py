"""
Database Migrations

This package contains database schema migrations for the code review system.
Each migration adds, modifies, or removes database tables and columns.

Migration Format:
- File naming: NNN_description.py (e.g., 001_add_suggestion_fields.py)
- Each migration should have upgrade() and downgrade() functions
- All migrations are designed to be idempotent when possible

Current Migrations:
- 001_add_suggestion_fields.py: Adds AI suggestion fields to Finding model

To apply migrations:
1. SQLite (automatic):
   python3 -c "from migrations.migrations_001 import execute_sqlite_migration; execute_sqlite_migration('codeproject.db')"

2. Using Alembic (when implemented):
   alembic upgrade head

3. Manual SQL:
   Execute the SQL statements in the migration file directly against your database
"""

import os
import sys
from pathlib import Path

# Get migrations directory
MIGRATIONS_DIR = Path(__file__).parent


def get_migrations() -> list:
    """Get list of migration files in order."""
    migration_files = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.py') and f[0].isdigit()])
    return migration_files


def apply_migration(migration_name: str, db_path: str = None):
    """
    Apply a specific migration.

    Args:
        migration_name: Name of the migration file (e.g., '001_add_suggestion_fields')
        db_path: Path to SQLite database (optional, defaults to codeproject.db)

    Returns:
        bool: True if successful, False otherwise
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(MIGRATIONS_DIR), 'codeproject.db')

    module_name = migration_name.replace('.py', '')

    try:
        # Import the migration module
        migration_module = __import__(f'migrations.{module_name}', fromlist=['execute_sqlite_migration'])

        if hasattr(migration_module, 'execute_sqlite_migration'):
            migration_module.execute_sqlite_migration(db_path)
            return True
        else:
            print(f"Migration {migration_name} does not have execute_sqlite_migration function")
            return False
    except Exception as e:
        print(f"Error applying migration {migration_name}: {e}")
        return False


def apply_all_migrations(db_path: str = None):
    """
    Apply all pending migrations in order.

    Args:
        db_path: Path to SQLite database (optional)

    Returns:
        dict: Results of each migration
    """
    results = {}
    migrations = get_migrations()

    print(f"Found {len(migrations)} migration(s)")

    for migration in migrations:
        print(f"\nApplying {migration}...")
        results[migration] = apply_migration(migration, db_path)

    return results


if __name__ == "__main__":
    # Allow running migrations from command line
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = None

    print("Running all migrations...")
    results = apply_all_migrations(db_path)

    # Print summary
    successful = sum(1 for v in results.values() if v)
    print(f"\n✓ Successfully applied {successful}/{len(results)} migrations")
