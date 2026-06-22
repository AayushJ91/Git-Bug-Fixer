"""
Feature Engineering Module.

Extracts handcrafted features from Pull Request data for baseline
ML models (Logistic Regression, XGBoost, etc.). These features are
also combined with CodeBERT embeddings in the hybrid model.

Features are grouped into categories:
1. Change Metrics — size and shape of the diff
2. Code Complexity — structural complexity signals
3. Text Features — commit messages and PR metadata
4. Entropy Metrics — distribution of changes across files
5. Historical Features — file and author history (requires DB)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from app.core.diff_parser import ParsedDiff

logger = structlog.get_logger(__name__)


@dataclass
class PRFeatures:
    """
    Complete feature vector for a Pull Request.

    All features are numerical (float) for direct use in ML models.
    Feature names follow sklearn convention: lowercase_with_underscores.
    """

    # --- Change Metrics ---
    num_files_changed: float = 0.0
    total_additions: float = 0.0
    total_deletions: float = 0.0
    total_lines_modified: float = 0.0
    max_file_additions: float = 0.0
    max_file_deletions: float = 0.0
    avg_file_additions: float = 0.0
    avg_file_deletions: float = 0.0
    num_hunks: float = 0.0
    addition_deletion_ratio: float = 0.5

    # --- File Type Distribution ---
    num_source_files: float = 0.0
    num_test_files: float = 0.0
    num_config_files: float = 0.0
    has_test_changes: float = 0.0  # bool as float (0/1)
    num_languages: float = 0.0

    # --- Code Complexity Signals ---
    max_hunk_size: float = 0.0
    avg_hunk_size: float = 0.0
    num_interleaved_changes: float = 0.0  # hunks with both adds and deletes

    # --- Text Features ---
    title_length: float = 0.0
    description_length: float = 0.0
    num_commits: float = 0.0
    avg_commit_message_length: float = 0.0
    has_issue_reference: float = 0.0  # bool as float
    has_breaking_keyword: float = 0.0  # bool as float

    # --- Entropy Metrics ---
    file_change_entropy: float = 0.0
    directory_entropy: float = 0.0

    # --- Historical Features (populated separately) ---
    max_file_bug_history: float = 0.0
    avg_file_bug_history: float = 0.0
    author_total_commits: float = 0.0
    author_bug_rate: float = 0.0
    max_file_num_authors: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for DataFrame / model input."""
        return {k: v for k, v in self.__dict__.items()}

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array(list(self.__dict__.values()), dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return list(self.__dict__.keys())


# --- Keywords and patterns ---
TEST_PATTERNS = re.compile(
    r"(test_|_test\.py|tests/|spec\.|\.spec\.|\.test\.|__tests__|test\.)",
    re.IGNORECASE,
)

CONFIG_PATTERNS = re.compile(
    r"(\.yml|\.yaml|\.toml|\.ini|\.cfg|\.conf|Dockerfile|docker-compose|Makefile|\.env)",
    re.IGNORECASE,
)

ISSUE_PATTERN = re.compile(r"#\d+|(?:fix|close|resolve)[sd]?\s+#\d+", re.IGNORECASE)

BREAKING_KEYWORDS = re.compile(
    r"\b(breaking|breaking.change|deprecat|remov|delet|drop.support)\b",
    re.IGNORECASE,
)


def _compute_entropy(values: list[float]) -> float:
    """
    Compute Shannon entropy of a distribution.

    High entropy = changes spread evenly across files (riskier).
    Low entropy = changes concentrated in few files (less risky).
    """
    if not values or sum(values) == 0:
        return 0.0

    total = sum(values)
    probs = [v / total for v in values if v > 0]

    return -sum(p * math.log2(p) for p in probs)


def extract_features(
    parsed_diff: ParsedDiff,
    pr_title: str = "",
    pr_description: str = "",
    commit_messages: list[str] | None = None,
) -> PRFeatures:
    """
    Extract all handcrafted features from a parsed diff and PR metadata.

    Args:
        parsed_diff: Structured diff data from diff_parser.
        pr_title: Pull Request title.
        pr_description: Pull Request description/body.
        commit_messages: List of commit messages.

    Returns:
        PRFeatures with all computed features.
    """
    features = PRFeatures()
    files = parsed_diff.files
    commit_messages = commit_messages or []

    if not files:
        return features

    # --- Change Metrics ---
    features.num_files_changed = float(len(files))
    features.total_additions = float(parsed_diff.total_additions)
    features.total_deletions = float(parsed_diff.total_deletions)
    features.total_lines_modified = float(parsed_diff.total_changes)

    file_additions = [f.additions for f in files]
    file_deletions = [f.deletions for f in files]

    features.max_file_additions = float(max(file_additions)) if file_additions else 0.0
    features.max_file_deletions = float(max(file_deletions)) if file_deletions else 0.0
    features.avg_file_additions = float(np.mean(file_additions)) if file_additions else 0.0
    features.avg_file_deletions = float(np.mean(file_deletions)) if file_deletions else 0.0

    total_hunks = sum(len(f.hunks) for f in files)
    features.num_hunks = float(total_hunks)

    if parsed_diff.total_changes > 0:
        features.addition_deletion_ratio = (
            parsed_diff.total_additions / parsed_diff.total_changes
        )

    # --- File Type Distribution ---
    languages = set()
    for f in files:
        if TEST_PATTERNS.search(f.file_path):
            features.num_test_files += 1
        elif CONFIG_PATTERNS.search(f.file_path):
            features.num_config_files += 1
        else:
            features.num_source_files += 1

        if f.language and f.language != "unknown":
            languages.add(f.language)

    features.has_test_changes = 1.0 if features.num_test_files > 0 else 0.0
    features.num_languages = float(len(languages))

    # --- Code Complexity Signals ---
    hunk_sizes = []
    interleaved = 0
    for f in files:
        for hunk in f.hunks:
            size = hunk.total_changes
            hunk_sizes.append(size)
            if hunk.added_lines and hunk.deleted_lines:
                interleaved += 1

    features.max_hunk_size = float(max(hunk_sizes)) if hunk_sizes else 0.0
    features.avg_hunk_size = float(np.mean(hunk_sizes)) if hunk_sizes else 0.0
    features.num_interleaved_changes = float(interleaved)

    # --- Text Features ---
    features.title_length = float(len(pr_title))
    features.description_length = float(len(pr_description))
    features.num_commits = float(len(commit_messages))

    if commit_messages:
        features.avg_commit_message_length = float(
            np.mean([len(m) for m in commit_messages])
        )

    # Check for issue references in title, description, and commits
    all_text = f"{pr_title} {pr_description} {' '.join(commit_messages)}"
    features.has_issue_reference = 1.0 if ISSUE_PATTERN.search(all_text) else 0.0
    features.has_breaking_keyword = 1.0 if BREAKING_KEYWORDS.search(all_text) else 0.0

    # --- Entropy Metrics ---
    # File-level entropy: how evenly are changes distributed?
    file_changes = [float(f.total_changes) for f in files]
    features.file_change_entropy = _compute_entropy(file_changes)

    # Directory-level entropy
    import os

    dir_changes: dict[str, float] = {}
    for f in files:
        dir_name = os.path.dirname(f.file_path) or "root"
        dir_changes[dir_name] = dir_changes.get(dir_name, 0.0) + f.total_changes
    features.directory_entropy = _compute_entropy(list(dir_changes.values()))

    logger.debug(
        "features_extracted",
        num_files=features.num_files_changed,
        total_changes=features.total_lines_modified,
        entropy=round(features.file_change_entropy, 3),
    )

    return features


def enrich_with_historical_features(
    features: PRFeatures,
    file_bug_counts: dict[str, int] | None = None,
    author_stats: dict[str, Any] | None = None,
    file_author_counts: dict[str, int] | None = None,
) -> PRFeatures:
    """
    Enrich features with historical data from the database.

    This is called separately because historical data requires
    database queries that may not be available during feature
    extraction.

    Args:
        features: Base features to enrich.
        file_bug_counts: {file_path: num_past_bugs} for each changed file.
        author_stats: {"total_commits": int, "bug_rate": float} for PR author.
        file_author_counts: {file_path: num_distinct_authors} for each file.
    """
    if file_bug_counts:
        bug_counts = list(file_bug_counts.values())
        features.max_file_bug_history = float(max(bug_counts)) if bug_counts else 0.0
        features.avg_file_bug_history = float(np.mean(bug_counts)) if bug_counts else 0.0

    if author_stats:
        features.author_total_commits = float(author_stats.get("total_commits", 0))
        features.author_bug_rate = float(author_stats.get("bug_rate", 0.0))

    if file_author_counts:
        counts = list(file_author_counts.values())
        features.max_file_num_authors = float(max(counts)) if counts else 0.0

    return features
