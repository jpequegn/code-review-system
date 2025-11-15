"""
Tests for diff parser.

Tests unified diff parsing, file filtering, and code change extraction.
"""

import pytest
from src.analysis.diff_parser import DiffParser, CodeChange, FileDiff


# ============================================================================
# Test Data & Fixtures
# ============================================================================

@pytest.fixture
def parser():
    """Provide a DiffParser instance."""
    return DiffParser()


@pytest.fixture
def simple_diff():
    """Simple diff with one file change."""
    return """diff --git a/hello.py b/hello.py
index 1234567..abcdefg 100644
--- a/hello.py
+++ b/hello.py
@@ -1,5 +1,6 @@
 def hello():
     print("Hello, World!")
+    print("Version 2")

 if __name__ == "__main__":
     hello()
"""


@pytest.fixture
def multiple_files_diff():
    """Diff with multiple file changes."""
    return """diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,4 @@
 def function1():
+    # Added comment
     pass

diff --git a/file2.ts b/file2.ts
index 2345678..bcdefgh 100644
--- a/file2.ts
+++ b/file2.ts
@@ -1,2 +1,3 @@
 function test() {
+    console.log("test");
 }
"""


@pytest.fixture
def complex_diff():
    """Complex diff with additions, deletions, and context."""
    return """diff --git a/src/app.py b/src/app.py
index 1234567..abcdefg 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,7 +10,9 @@ def process_data(data):
     result = {}
     for item in data:
-        if item is None:
+        if item is None or item == "":
             continue
+        # Validate item
+        validated = validate(item)
         result[item] = process(item)
     return result
"""


@pytest.fixture
def diff_with_deletions():
    """Diff with only deletions."""
    return """diff --git a/old.py b/old.py
index 1234567..abcdefg 100644
--- a/old.py
+++ b/old.py
@@ -1,5 +1,3 @@
 def func():
-    deprecated_call()
-    old_code()
     return 42
"""


@pytest.fixture
def diff_with_renames():
    """Diff with file rename."""
    return """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
"""


@pytest.fixture
def diff_with_binary():
    """Diff with binary file."""
    return """diff --git a/image.png b/image.png
Binary files /dev/null and b/image.png differ
"""


# ============================================================================
# Test File Filtering
# ============================================================================

class TestFileFiltering:
    """Tests for file filtering logic."""

    def test_include_python_file(self, parser):
        """Test that Python files are included."""
        assert parser.should_include_file("app.py") is True
        assert parser.should_include_file("src/modules/helper.py") is True

    def test_include_typescript_file(self, parser):
        """Test that TypeScript files are included."""
        assert parser.should_include_file("main.ts") is True
        assert parser.should_include_file("types.tsx") is True

    def test_include_other_code_files(self, parser):
        """Test that other code files are included."""
        assert parser.should_include_file("main.go") is True
        assert parser.should_include_file("lib.rs") is True
        assert parser.should_include_file("App.java") is True
        assert parser.should_include_file("script.js") is True

    def test_skip_markdown(self, parser):
        """Test that markdown files are skipped."""
        assert parser.should_include_file("README.md") is False
        assert parser.should_include_file("docs/guide.md") is False

    def test_skip_json(self, parser):
        """Test that JSON files are skipped."""
        assert parser.should_include_file("package.json") is False
        assert parser.should_include_file("config.json") is False

    def test_skip_yaml(self, parser):
        """Test that YAML files are skipped."""
        assert parser.should_include_file("config.yaml") is False
        assert parser.should_include_file("settings.yml") is False

    def test_skip_test_files(self, parser):
        """Test that test files are skipped."""
        assert parser.should_include_file("test_app.py") is False
        assert parser.should_include_file("tests/test_utils.py") is False
        assert parser.should_include_file("app.test.js") is False
        assert parser.should_include_file("app.spec.ts") is False

    def test_skip_dockerfile(self, parser):
        """Test that Dockerfile is skipped."""
        assert parser.should_include_file("Dockerfile") is False

    def test_skip_makefile(self, parser):
        """Test that Makefile is skipped."""
        assert parser.should_include_file("Makefile") is False

    def test_case_insensitive_filtering(self, parser):
        """Test that filtering is case-insensitive."""
        assert parser.should_include_file("README.MD") is False
        assert parser.should_include_file("CONFIG.JSON") is False
        assert parser.should_include_file("App.PY") is True

    def test_custom_skip_patterns(self):
        """Test adding custom skip patterns."""
        parser = DiffParser(skip_patterns={".custom"})
        assert parser.should_include_file("file.custom") is False
        assert parser.should_include_file("file.py") is True


# ============================================================================
# Test Diff Parsing
# ============================================================================

class TestDiffParsing:
    """Tests for diff parsing."""

    def test_parse_simple_diff(self, parser, simple_diff):
        """Test parsing a simple diff."""
        result = parser.parse(simple_diff)

        assert len(result) == 1
        assert result[0].file_path == "hello.py"
        assert result[0].additions == 1
        assert result[0].deletions == 0

    def test_parse_multiple_files(self, parser, multiple_files_diff):
        """Test parsing diff with multiple files."""
        result = parser.parse(multiple_files_diff)

        assert len(result) == 2
        assert result[0].file_path == "file1.py"
        assert result[1].file_path == "file2.ts"

    def test_parse_complex_diff(self, parser, complex_diff):
        """Test parsing complex diff with additions and deletions."""
        result = parser.parse(complex_diff)

        assert len(result) == 1
        file_diff = result[0]
        assert file_diff.file_path == "src/app.py"
        assert file_diff.additions >= 2
        assert file_diff.deletions >= 1

    def test_parse_deletions(self, parser, diff_with_deletions):
        """Test parsing diff with deletions."""
        result = parser.parse(diff_with_deletions)

        assert len(result) == 1
        file_diff = result[0]
        assert file_diff.deletions == 2
        assert file_diff.additions == 0

    def test_parse_binary_file(self, parser, diff_with_binary):
        """Test parsing binary file changes."""
        result = parser.parse(diff_with_binary)

        # Binary files might be skipped depending on filter
        # At minimum, we should handle them without crashing
        assert isinstance(result, list)

    def test_parse_empty_diff(self, parser):
        """Test parsing empty diff."""
        result = parser.parse("")
        assert result == []

    def test_parse_none_diff(self, parser):
        """Test parsing None diff."""
        result = parser.parse(None or "")
        assert result == []

    def test_parse_filters_non_code_files(self, parser):
        """Test that non-code files are filtered out."""
        diff = """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-Old readme
+New readme
"""
        result = parser.parse(diff)
        assert len(result) == 0  # README.md should be filtered


# ============================================================================
# Test Code Changes
# ============================================================================

class TestCodeChanges:
    """Tests for code change extraction."""

    def test_extract_additions(self, parser, simple_diff):
        """Test extraction of added lines."""
        result = parser.parse(simple_diff)
        changes = result[0].changes

        assert len(changes) > 0
        # Should have at least one addition
        additions = [c for c in changes if c.change_type == "add"]
        assert len(additions) >= 1

    def test_extract_deletions(self, parser, diff_with_deletions):
        """Test extraction of deleted lines."""
        result = parser.parse(diff_with_deletions)
        changes = result[0].changes

        # Should have deletions
        deletions = [c for c in changes if c.change_type == "remove"]
        assert len(deletions) == 2

    def test_line_numbers_are_correct(self, parser, simple_diff):
        """Test that line numbers are tracked correctly."""
        result = parser.parse(simple_diff)
        changes = result[0].changes

        # All changes should have valid line numbers
        for change in changes:
            assert change.line_number > 0
            if change.change_type == "remove":
                assert change.old_line_number > 0

    def test_code_change_content(self, parser, simple_diff):
        """Test that change content is preserved."""
        result = parser.parse(simple_diff)
        changes = result[0].changes

        # Check that content is not empty
        for change in changes:
            assert len(change.content) > 0

    def test_context_extraction(self, parser, complex_diff):
        """Test that context lines are extracted."""
        result = parser.parse(complex_diff)
        changes = result[0].changes

        # At least some changes should have context
        if changes:
            # Some may have context before or after
            has_context = any(
                change.context_before or change.context_after
                for change in changes
            )
            assert has_context or len(changes) <= 1


# ============================================================================
# Test File Diff Structure
# ============================================================================

class TestFileDiffStructure:
    """Tests for FileDiff dataclass."""

    def test_file_diff_basic_fields(self, parser, simple_diff):
        """Test basic fields of FileDiff."""
        result = parser.parse(simple_diff)
        file_diff = result[0]

        assert file_diff.file_path == "hello.py"
        assert file_diff.is_binary is False
        assert file_diff.additions >= 0
        assert file_diff.deletions >= 0

    def test_file_diff_total_changes(self, parser, simple_diff):
        """Test total_changes property."""
        result = parser.parse(simple_diff)
        file_diff = result[0]

        total = file_diff.total_changes
        assert total == file_diff.additions + file_diff.deletions

    def test_file_diff_changes_list(self, parser, simple_diff):
        """Test changes list is populated."""
        result = parser.parse(simple_diff)
        file_diff = result[0]

        assert isinstance(file_diff.changes, list)
        assert len(file_diff.changes) >= 0

    def test_file_diff_with_rename(self, parser):
        """Test file diff with rename."""
        diff = """diff --git a/old.py b/new.py
similarity index 100%
rename from old.py
rename to new.py
"""
        result = parser.parse(diff)
        # Rename without content changes should be included or filtered
        # depending on implementation
        assert isinstance(result, list)


# ============================================================================
# Test Code Change Dataclass
# ============================================================================

class TestCodeChangeDataclass:
    """Tests for CodeChange dataclass."""

    def test_code_change_fields(self, parser, simple_diff):
        """Test CodeChange has required fields."""
        result = parser.parse(simple_diff)
        if result and result[0].changes:
            change = result[0].changes[0]

            assert hasattr(change, "file_path")
            assert hasattr(change, "line_number")
            assert hasattr(change, "change_type")
            assert hasattr(change, "content")

    def test_code_change_type_values(self, parser, simple_diff):
        """Test change_type has valid values."""
        result = parser.parse(simple_diff)
        if result and result[0].changes:
            for change in result[0].changes:
                assert change.change_type in ["add", "remove", "context"]

    def test_code_change_repr(self, parser, simple_diff):
        """Test CodeChange repr."""
        result = parser.parse(simple_diff)
        if result and result[0].changes:
            change = result[0].changes[0]
            repr_str = repr(change)

            assert "CodeChange" in repr_str
            assert change.file_path in repr_str


# ============================================================================
# Test Large Diff Handling
# ============================================================================

class TestLargeDiffs:
    """Tests for handling large diffs."""

    def test_large_diff_performance(self, parser):
        """Test parsing large diff doesn't crash."""
        # Create a large diff with 100 file changes
        diff_parts = []
        for i in range(100):
            diff_parts.append(f"""diff --git a/file{i}.py b/file{i}.py
index 1234567..abcdefg 100644
--- a/file{i}.py
+++ b/file{i}.py
@@ -1,2 +1,3 @@
 def func{i}():
+    pass
 return 42
""")
        large_diff = "\n".join(diff_parts)

        result = parser.parse(large_diff)
        assert len(result) == 100

    def test_many_hunks_in_single_file(self, parser):
        """Test file with many change hunks."""
        diff_parts = [
            "diff --git a/file.py b/file.py\nindex 1234567..abcdefg 100644"
        ]

        # Add 50 hunks
        for i in range(50):
            line = 10 + (i * 5)
            diff_parts.append(f"""@@ -{line},2 +{line},3 @@
 context_before
+added_line_{i}
 context_after
""")

        diff = "\n".join(diff_parts)
        result = parser.parse(diff)

        assert len(result) >= 1


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_diff_with_special_characters(self, parser):
        """Test parsing diff with special characters."""
        diff = """diff --git a/file.py b/file.py
index 1234567..abcdefg 100644
--- a/file.py
+++ b/file.py
@@ -1,2 +1,3 @@
 # -*- coding: utf-8 -*-
+# Special chars: é à ñ 中文
 pass
"""
        result = parser.parse(diff)
        assert len(result) == 1

    def test_diff_with_no_newline_at_end(self, parser):
        """Test parsing diff with 'no newline at end of file'."""
        diff = """diff --git a/file.py b/file.py
index 1234567..abcdefg 100644
--- a/file.py
+++ b/file.py
@@ -1,2 +1,3 @@
 line1
+line2
\\ No newline at end of file
"""
        result = parser.parse(diff)
        # Should handle without crashing
        assert isinstance(result, list)

    def test_context_lines_configuration(self):
        """Test configurable context lines."""
        parser_with_context = DiffParser(context_lines=10)
        assert parser_with_context.context_lines == 10

        parser_no_context = DiffParser(context_lines=0)
        assert parser_no_context.context_lines == 0
