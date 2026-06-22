"""
Diff Parser Module.

Parses unified diff text (as returned by GitHub's API) into
structured representations suitable for feature extraction
and model input tokenization.

Uses the `unidiff` library for robust diff parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None  # type: ignore[assignment, misc]

logger = structlog.get_logger(__name__)


class ChangeType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    RENAME = "rename"


@dataclass
class DiffHunk:
    """A single hunk (contiguous change block) within a file diff."""

    source_start: int
    source_length: int
    target_start: int
    target_length: int
    added_lines: list[str] = field(default_factory=list)
    deleted_lines: list[str] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.added_lines) + len(self.deleted_lines)


@dataclass
class FileDiff:
    """Parsed diff for a single file."""

    file_path: str
    old_path: str = ""
    change_type: ChangeType = ChangeType.MODIFY
    language: str = ""
    additions: int = 0
    deletions: int = 0
    hunks: list[DiffHunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions

    @property
    def change_ratio(self) -> float:
        """Ratio of additions to total changes. 0.5 = balanced, 1.0 = all additions."""
        if self.total_changes == 0:
            return 0.5
        return self.additions / self.total_changes

    def get_flattened_diff(self, max_lines: int = 500) -> str:
        """
        Get a flattened text representation of the diff for model input.

        Format: +added_line / -deleted_line, preserving order.
        """
        lines: list[str] = []
        for hunk in self.hunks:
            for line in hunk.added_lines:
                lines.append(f"+ {line}")
                if len(lines) >= max_lines:
                    return "\n".join(lines)
            for line in hunk.deleted_lines:
                lines.append(f"- {line}")
                if len(lines) >= max_lines:
                    return "\n".join(lines)
        return "\n".join(lines)


@dataclass
class ParsedDiff:
    """Complete parsed diff for an entire Pull Request."""

    files: list[FileDiff] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    total_files: int = 0

    @property
    def total_changes(self) -> int:
        return self.total_additions + self.total_deletions

    @property
    def file_paths(self) -> list[str]:
        return [f.file_path for f in self.files]

    def get_files_by_language(self, language: str) -> list[FileDiff]:
        return [f for f in self.files if f.language == language]


# --- Language detection by file extension ---
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
}

# File patterns to skip (non-code files)
SKIP_PATTERNS = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "go.sum",
    "Cargo.lock",
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "CHANGELOG.md",
}


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    import os

    _, ext = os.path.splitext(file_path.lower())
    return LANGUAGE_MAP.get(ext, "unknown")


def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped (lock files, binaries, etc.)."""
    import os

    basename = os.path.basename(file_path)
    if basename in SKIP_PATTERNS:
        return True
    # Skip hidden files/directories
    if any(part.startswith(".") for part in file_path.split("/")):
        return True
    return False


def parse_diff(diff_text: str) -> ParsedDiff:
    """
    Parse a unified diff string into structured FileDiff objects.

    Args:
        diff_text: Unified diff text (e.g., from GitHub API).

    Returns:
        ParsedDiff with structured file-level and hunk-level data.
    """
    if PatchSet is None:
        logger.warning("unidiff_not_installed", fallback="manual_parse")
        return _parse_diff_manual(diff_text)

    try:
        patch = PatchSet(diff_text)
    except Exception as e:
        logger.warning("diff_parse_error", error=str(e), fallback="manual_parse")
        return _parse_diff_manual(diff_text)

    parsed = ParsedDiff()

    for patched_file in patch:
        file_path = patched_file.path
        if should_skip_file(file_path):
            continue

        # Determine change type
        if patched_file.is_added_file:
            change_type = ChangeType.ADD
        elif patched_file.is_removed_file:
            change_type = ChangeType.DELETE
        elif patched_file.is_rename:
            change_type = ChangeType.RENAME
        else:
            change_type = ChangeType.MODIFY

        file_diff = FileDiff(
            file_path=file_path,
            old_path=patched_file.source_file or "",
            change_type=change_type,
            language=detect_language(file_path),
            additions=patched_file.added,
            deletions=patched_file.removed,
            is_binary=patched_file.is_binary_file,
        )

        # Parse hunks
        for hunk in patched_file:
            diff_hunk = DiffHunk(
                source_start=hunk.source_start,
                source_length=hunk.source_length,
                target_start=hunk.target_start,
                target_length=hunk.target_length,
            )

            for line in hunk:
                if line.is_added:
                    diff_hunk.added_lines.append(line.value.rstrip("\n"))
                elif line.is_removed:
                    diff_hunk.deleted_lines.append(line.value.rstrip("\n"))
                else:
                    diff_hunk.context_lines.append(line.value.rstrip("\n"))

            file_diff.hunks.append(diff_hunk)

        parsed.files.append(file_diff)
        parsed.total_additions += file_diff.additions
        parsed.total_deletions += file_diff.deletions

    parsed.total_files = len(parsed.files)

    logger.info(
        "diff_parsed",
        total_files=parsed.total_files,
        total_additions=parsed.total_additions,
        total_deletions=parsed.total_deletions,
    )

    return parsed


def _parse_diff_manual(diff_text: str) -> ParsedDiff:
    """
    Fallback manual diff parser when unidiff is not available.

    Handles basic unified diff format.
    """
    parsed = ParsedDiff()
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            # New file
            if current_file:
                if current_hunk:
                    current_file.hunks.append(current_hunk)
                parsed.files.append(current_file)

            # Extract file path from "diff --git a/path b/path"
            parts = line.split(" b/", 1)
            file_path = parts[1] if len(parts) > 1 else "unknown"

            if should_skip_file(file_path):
                current_file = None
                current_hunk = None
                continue

            current_file = FileDiff(
                file_path=file_path,
                language=detect_language(file_path),
            )
            current_hunk = None

        elif line.startswith("@@") and current_file:
            # New hunk
            if current_hunk:
                current_file.hunks.append(current_hunk)
            current_hunk = DiffHunk(
                source_start=0,
                source_length=0,
                target_start=0,
                target_length=0,
            )

        elif current_file and current_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.added_lines.append(line[1:])
                current_file.additions += 1
                parsed.total_additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.deleted_lines.append(line[1:])
                current_file.deletions += 1
                parsed.total_deletions += 1
            else:
                current_hunk.context_lines.append(line)

    # Don't forget the last file
    if current_file:
        if current_hunk:
            current_file.hunks.append(current_hunk)
        parsed.files.append(current_file)

    parsed.total_files = len(parsed.files)
    return parsed


def create_model_input(
    parsed_diff: ParsedDiff,
    pr_title: str = "",
    pr_description: str = "",
    commit_messages: list[str] | None = None,
    max_diff_lines: int = 400,
) -> str:
    """
    Create the flattened text input for the ML model.

    Format: [CLS] <metadata> [SEP] <diff_tokens> [SEP]

    The actual [CLS]/[SEP] tokens are added by the tokenizer.
    This function produces the raw text.
    """
    # Part 1: Natural language metadata
    metadata_parts = []
    if pr_title:
        metadata_parts.append(f"Title: {pr_title}")
    if pr_description:
        # Truncate long descriptions
        desc = pr_description[:500]
        metadata_parts.append(f"Description: {desc}")
    if commit_messages:
        msgs = " | ".join(commit_messages[:5])  # Max 5 commit messages
        metadata_parts.append(f"Commits: {msgs}")

    metadata = " ".join(metadata_parts)

    # Part 2: Flattened diff
    diff_lines: list[str] = []
    remaining_lines = max_diff_lines

    for file_diff in parsed_diff.files:
        if remaining_lines <= 0:
            break

        diff_lines.append(f"[FILE] {file_diff.file_path}")
        remaining_lines -= 1

        for hunk in file_diff.hunks:
            for added in hunk.added_lines:
                if remaining_lines <= 0:
                    break
                diff_lines.append(f"+ {added}")
                remaining_lines -= 1

            for deleted in hunk.deleted_lines:
                if remaining_lines <= 0:
                    break
                diff_lines.append(f"- {deleted}")
                remaining_lines -= 1

    diff_text = "\n".join(diff_lines)

    # Combine: metadata [SEP] diff
    return f"{metadata} [SEP] {diff_text}"
