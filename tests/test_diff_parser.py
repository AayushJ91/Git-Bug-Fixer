"""
Tests for the diff parser module.

Tests cover:
- Unified diff parsing (unidiff library)
- Manual fallback parser
- Language detection
- File skipping logic
- Model input creation
"""

from __future__ import annotations

import pytest

from app.core.diff_parser import (
    ChangeType,
    FileDiff,
    ParsedDiff,
    create_model_input,
    detect_language,
    parse_diff,
    should_skip_file,
)


# --- Sample Diffs ---

SAMPLE_DIFF = """\
diff --git a/src/database.py b/src/database.py
index abc1234..def5678 100644
--- a/src/database.py
+++ b/src/database.py
@@ -10,6 +10,8 @@ class Database:
     def connect(self):
         self.conn = psycopg2.connect(self.url)
+        self.conn.autocommit = True
+        self.pool = ConnectionPool(self.conn)
         return self.conn

@@ -25,3 +27,5 @@ class Database:
     def query(self, sql):
-        cursor = self.conn.cursor()
+        cursor = self.pool.get_cursor()
+        cursor.execute(sql)
         return cursor
"""

MULTI_FILE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index 111..222 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,4 @@
+import jwt
 from flask import request

 def authenticate(token):
@@ -5,2 +6,3 @@ def authenticate(token):
-    return verify(token)
+    decoded = jwt.decode(token, SECRET_KEY)
+    return decoded

diff --git a/tests/test_auth.py b/tests/test_auth.py
index 333..444 100644
--- a/tests/test_auth.py
+++ b/tests/test_auth.py
@@ -1,3 +1,5 @@
+import pytest
+
 def test_auth():
     assert authenticate("valid") is not None
"""


class TestLanguageDetection:
    def test_python(self) -> None:
        assert detect_language("src/main.py") == "python"

    def test_javascript(self) -> None:
        assert detect_language("index.js") == "javascript"

    def test_typescript(self) -> None:
        assert detect_language("app.tsx") == "typescript"

    def test_java(self) -> None:
        assert detect_language("Main.java") == "java"

    def test_unknown(self) -> None:
        assert detect_language("Makefile") == "unknown"

    def test_nested_path(self) -> None:
        assert detect_language("src/components/Button.tsx") == "typescript"


class TestFileSkipping:
    def test_lock_file(self) -> None:
        assert should_skip_file("package-lock.json") is True

    def test_yarn_lock(self) -> None:
        assert should_skip_file("yarn.lock") is True

    def test_source_file(self) -> None:
        assert should_skip_file("src/main.py") is False

    def test_gitignore(self) -> None:
        assert should_skip_file(".gitignore") is True


class TestDiffParsing:
    def test_parse_single_file(self) -> None:
        result = parse_diff(SAMPLE_DIFF)

        assert isinstance(result, ParsedDiff)
        assert result.total_files == 1
        assert result.files[0].file_path == "src/database.py"
        assert result.files[0].language == "python"
        assert result.files[0].additions > 0

    def test_parse_multi_file(self) -> None:
        result = parse_diff(MULTI_FILE_DIFF)

        assert result.total_files == 2
        paths = [f.file_path for f in result.files]
        assert "src/auth.py" in paths
        assert "tests/test_auth.py" in paths

    def test_parse_empty_diff(self) -> None:
        result = parse_diff("")

        assert result.total_files == 0
        assert result.total_additions == 0
        assert result.total_deletions == 0

    def test_hunks_extracted(self) -> None:
        result = parse_diff(SAMPLE_DIFF)
        file_diff = result.files[0]

        assert len(file_diff.hunks) >= 1
        assert file_diff.hunks[0].added_lines  # Should have added lines

    def test_total_changes(self) -> None:
        result = parse_diff(SAMPLE_DIFF)

        assert result.total_changes == result.total_additions + result.total_deletions
        assert result.total_changes > 0


class TestModelInput:
    def test_create_model_input(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        text = create_model_input(
            parsed,
            pr_title="Fix database connection pooling",
            pr_description="Adds connection pooling to prevent connection exhaustion.",
            commit_messages=["fix: add connection pool"],
        )

        assert "Fix database connection pooling" in text
        assert "[SEP]" in text
        assert "[FILE]" in text
        assert "database.py" in text

    def test_model_input_truncation(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        text = create_model_input(parsed, max_diff_lines=5)

        # Should be truncated
        lines = text.split("[SEP]")[1].strip().split("\n")
        assert len(lines) <= 6  # 5 lines + file header


class TestFileDiff:
    def test_change_ratio_balanced(self) -> None:
        fd = FileDiff(file_path="test.py", additions=5, deletions=5)
        assert fd.change_ratio == 0.5

    def test_change_ratio_all_additions(self) -> None:
        fd = FileDiff(file_path="test.py", additions=10, deletions=0)
        assert fd.change_ratio == 1.0

    def test_change_ratio_no_changes(self) -> None:
        fd = FileDiff(file_path="test.py", additions=0, deletions=0)
        assert fd.change_ratio == 0.5
