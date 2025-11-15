"""
Diff Parser for Code Analysis

Parses unified diff format from git and extracts code changes.
Filters to relevant code files and provides structured access to changes.
"""

import re
import logging
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# File Patterns for Filtering
# ============================================================================

# Code file extensions to include
CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",  # Python, TypeScript, JavaScript
    ".go", ".rs", ".java",  # Go, Rust, Java
    ".cpp", ".c", ".cc", ".h", ".hpp",  # C/C++
    ".rb", ".php", ".cs", ".kt", ".scala",  # Ruby, PHP, C#, Kotlin, Scala
    ".swift", ".m", ".mm",  # Swift, Objective-C
    ".r", ".sql",  # R, SQL
}

# Patterns to skip (paths/extensions)
SKIP_PATTERNS = {
    # Test files
    "test_", "tests/", "_test.", ".test.", ".spec.",
    # Documentation
    ".md", ".txt", ".rst",
    # Configuration and metadata
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".html", ".css", ".scss", ".sass", ".less",
    # Build and package files
    "Dockerfile", "Makefile", ".lock", ".gradle", ".pom",
    # Source control and git
    ".git", ".gitignore",
    # Compiled and binary files
    ".pyc", ".o", ".so", ".a", ".lib", ".dll", ".exe",
    ".class", ".jar", ".war",
}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CodeChange:
    """Represents a single code change (added/removed/modified line)."""
    file_path: str  # Path to the file
    line_number: int  # Line number in the new file
    old_line_number: Optional[int]  # Line number in old file (for context)
    change_type: str  # "add" (+), "remove" (-), "context" ( )
    content: str  # The actual line content
    context_before: List[str] = field(default_factory=list)  # Lines before
    context_after: List[str] = field(default_factory=list)  # Lines after

    def __repr__(self) -> str:
        return (
            f"CodeChange({self.file_path}:{self.line_number} "
            f"[{self.change_type}] {self.content[:50]}...)"
        )


@dataclass
class FileDiff:
    """Represents all changes in a single file."""
    file_path: str  # Path to the file
    old_path: Optional[str]  # Path in old version (for renames)
    is_binary: bool  # Whether file is binary
    additions: int  # Number of added lines
    deletions: int  # Number of deleted lines
    changes: List[CodeChange] = field(default_factory=list)  # All changes

    @property
    def total_changes(self) -> int:
        """Total number of changed lines."""
        return self.additions + self.deletions

    def __repr__(self) -> str:
        return (
            f"FileDiff({self.file_path} "
            f"+{self.additions}/-{self.deletions} "
            f"{len(self.changes)} changes)"
        )


# ============================================================================
# Diff Parser
# ============================================================================

class DiffParser:
    """
    Parser for unified diff format (git diff output).

    Extracts code changes from diffs, filters to relevant files,
    and provides structured access to changed code.
    """

    def __init__(
        self,
        context_lines: int = 5,
        skip_patterns: Optional[Set[str]] = None,
    ):
        """
        Initialize DiffParser.

        Args:
            context_lines: Number of context lines to include around changes
            skip_patterns: Additional patterns to skip (merged with defaults)
        """
        self.context_lines = context_lines
        self.skip_patterns = SKIP_PATTERNS.copy()
        if skip_patterns:
            self.skip_patterns.update(skip_patterns)

    def should_include_file(self, file_path: str) -> bool:
        """
        Determine if a file should be included in analysis.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be included, False otherwise
        """
        # Skip if matches any skip pattern
        lower_path = file_path.lower()
        for pattern in self.skip_patterns:
            if pattern in lower_path or lower_path.endswith(pattern):
                return False

        # Include if has code file extension
        for ext in CODE_EXTENSIONS:
            if lower_path.endswith(ext):
                return True

        return False

    def parse(self, diff_text: str) -> List[FileDiff]:
        """
        Parse unified diff format and extract file diffs.

        Args:
            diff_text: Unified diff output (e.g., from git diff)

        Returns:
            List of FileDiff objects for all changed files

        Raises:
            ValueError: If diff format is invalid
        """
        if not diff_text:
            return []

        file_diffs: List[FileDiff] = []
        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            # Look for file header (diff --git a/... b/...)
            if lines[i].startswith("diff --git"):
                file_diff = self._parse_file_diff(lines, i)
                i = file_diff["end_index"]

                # Only include if matches filter
                if self.should_include_file(file_diff["file_path"]):
                    file_diffs.append(file_diff["diff"])
            else:
                i += 1

        return file_diffs

    def _parse_file_diff(
        self,
        lines: List[str],
        start_index: int
    ) -> Dict:
        """
        Parse a single file's diff section.

        Args:
            lines: All lines in the diff
            start_index: Index of the "diff --git" line

        Returns:
            Dictionary with file_path, diff object, and end_index
        """
        diff_line = lines[start_index]

        # Extract file paths from "diff --git a/path b/path"
        parts = diff_line.split()
        if len(parts) < 4:
            logger.warning(f"Invalid diff header: {diff_line}")
            return {"file_path": "", "diff": None, "end_index": start_index + 1}

        old_path = parts[2][2:]  # Remove "a/"
        new_path = parts[3][2:]  # Remove "b/"

        file_path = new_path if new_path != "/dev/null" else old_path

        # Initialize file diff
        file_diff = FileDiff(
            file_path=file_path,
            old_path=old_path if old_path != "/dev/null" else None,
            is_binary=False,
            additions=0,
            deletions=0,
        )

        i = start_index + 1

        # Parse metadata lines
        while i < len(lines):
            line = lines[i]

            # Check for binary file marker
            if "Binary files" in line:
                file_diff.is_binary = True
                i += 1
                break

            # Look for hunk header (@@...)
            if line.startswith("@@"):
                break

            i += 1

        # Parse hunks
        while i < len(lines):
            if lines[i].startswith("diff --git"):
                # Next file
                break

            if lines[i].startswith("@@"):
                i = self._parse_hunk(lines, i, file_diff)
            else:
                i += 1

        return {
            "file_path": file_path,
            "diff": file_diff,
            "end_index": i,
        }

    def _parse_hunk(
        self,
        lines: List[str],
        start_index: int,
        file_diff: FileDiff
    ) -> int:
        """
        Parse a single hunk (@@...) section.

        Args:
            lines: All lines in the diff
            start_index: Index of the "@@" line
            file_diff: FileDiff object to populate

        Returns:
            Index of the next line to process
        """
        hunk_header = lines[start_index]

        # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
        # Example: @@ -1,10 +1,12 @@
        match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', hunk_header)
        if not match:
            logger.warning(f"Invalid hunk header: {hunk_header}")
            return start_index + 1

        old_start = int(match.group(1))
        new_start = int(match.group(3))

        old_line_num = old_start
        new_line_num = new_start

        i = start_index + 1
        hunk_lines = []

        # Process lines in the hunk
        while i < len(lines):
            line = lines[i]

            # End of hunk
            if line.startswith("@@") or line.startswith("diff --git"):
                break

            # Skip lines without change indicators
            if not line or (
                not line.startswith("+")
                and not line.startswith("-")
                and not line.startswith(" ")
                and not line.startswith("\\")
            ):
                if line.startswith("---") or line.startswith("+++"):
                    i += 1
                    continue
                # Unclear line, might be malformed
                i += 1
                continue

            # Handle "\ No newline at end of file"
            if line.startswith("\\"):
                i += 1
                continue

            # Extract change type and content
            if line.startswith("+"):
                change_type = "add"
                content = line[1:]
                hunk_lines.append({
                    "type": change_type,
                    "new_line": new_line_num,
                    "old_line": None,
                    "content": content,
                })
                file_diff.additions += 1
                new_line_num += 1

            elif line.startswith("-"):
                change_type = "remove"
                content = line[1:]
                hunk_lines.append({
                    "type": change_type,
                    "new_line": None,
                    "old_line": old_line_num,
                    "content": content,
                })
                file_diff.deletions += 1
                old_line_num += 1

            else:  # " " - context line
                hunk_lines.append({
                    "type": "context",
                    "new_line": new_line_num,
                    "old_line": old_line_num,
                    "content": line[1:],
                })
                old_line_num += 1
                new_line_num += 1

            i += 1

        # Create CodeChange objects with context
        for idx, hunk_line in enumerate(hunk_lines):
            # Only create CodeChange for actual changes (not context)
            if hunk_line["type"] in ["add", "remove"]:
                # Gather context lines
                context_before = []
                context_after = []

                # Context before
                for j in range(max(0, idx - self.context_lines), idx):
                    if hunk_lines[j]["type"] == "context":
                        context_before.append(hunk_lines[j]["content"])

                # Context after
                for j in range(idx + 1, min(len(hunk_lines), idx + self.context_lines + 1)):
                    if hunk_lines[j]["type"] == "context":
                        context_after.append(hunk_lines[j]["content"])

                code_change = CodeChange(
                    file_path=file_diff.file_path,
                    line_number=hunk_line["new_line"] or hunk_line["old_line"],
                    old_line_number=hunk_line["old_line"],
                    change_type=hunk_line["type"],
                    content=hunk_line["content"],
                    context_before=context_before,
                    context_after=context_after,
                )
                file_diff.changes.append(code_change)

        return i
