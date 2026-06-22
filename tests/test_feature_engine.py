"""
Tests for the feature engineering module.
"""

from __future__ import annotations

import pytest
import numpy as np

from app.core.diff_parser import DiffHunk, FileDiff, ParsedDiff, ChangeType
from app.core.feature_engine import (
    PRFeatures,
    extract_features,
    enrich_with_historical_features,
    _compute_entropy,
)


def _make_parsed_diff(
    num_files: int = 3,
    additions_per_file: int = 20,
    deletions_per_file: int = 10,
    hunks_per_file: int = 2,
) -> ParsedDiff:
    """Create a synthetic ParsedDiff for testing."""
    files = []
    for i in range(num_files):
        hunks = []
        for j in range(hunks_per_file):
            hunks.append(
                DiffHunk(
                    source_start=1 + j * 10,
                    source_length=10,
                    target_start=1 + j * 10,
                    target_length=12,
                    added_lines=[f"    new_code_{k}()" for k in range(additions_per_file // hunks_per_file)],
                    deleted_lines=[f"    old_code_{k}()" for k in range(deletions_per_file // hunks_per_file)],
                    context_lines=["    # context"],
                )
            )

        files.append(
            FileDiff(
                file_path=f"src/module_{i}.py",
                change_type=ChangeType.MODIFY,
                language="python",
                additions=additions_per_file,
                deletions=deletions_per_file,
                hunks=hunks,
            )
        )

    return ParsedDiff(
        files=files,
        total_additions=additions_per_file * num_files,
        total_deletions=deletions_per_file * num_files,
        total_files=num_files,
    )


class TestEntropy:
    def test_zero_entropy(self) -> None:
        """Single file = zero entropy."""
        assert _compute_entropy([100.0]) == 0.0

    def test_max_entropy(self) -> None:
        """Equal distribution = max entropy."""
        entropy = _compute_entropy([10.0, 10.0, 10.0, 10.0])
        assert entropy == pytest.approx(2.0, abs=0.01)

    def test_empty(self) -> None:
        assert _compute_entropy([]) == 0.0


class TestExtractFeatures:
    def test_basic_features(self) -> None:
        diff = _make_parsed_diff(num_files=3, additions_per_file=20, deletions_per_file=10)
        features = extract_features(diff, pr_title="Fix bug in module", pr_description="Fixes #42")

        assert features.num_files_changed == 3
        assert features.total_additions == 60
        assert features.total_deletions == 30
        assert features.total_lines_modified == 90

    def test_test_file_detection(self) -> None:
        diff = ParsedDiff(
            files=[
                FileDiff(file_path="src/main.py", language="python", additions=10, deletions=5),
                FileDiff(file_path="tests/test_main.py", language="python", additions=5, deletions=0),
            ],
            total_additions=15,
            total_deletions=5,
            total_files=2,
        )
        features = extract_features(diff)

        assert features.has_test_changes == 1.0
        assert features.num_test_files == 1
        assert features.num_source_files == 1

    def test_issue_reference(self) -> None:
        diff = _make_parsed_diff(num_files=1)
        features = extract_features(
            diff,
            pr_title="Fix #123",
            pr_description="Resolves the issue",
        )
        assert features.has_issue_reference == 1.0

    def test_no_issue_reference(self) -> None:
        diff = _make_parsed_diff(num_files=1)
        features = extract_features(diff, pr_title="Update stuff")
        assert features.has_issue_reference == 0.0

    def test_breaking_keyword(self) -> None:
        diff = _make_parsed_diff(num_files=1)
        features = extract_features(
            diff,
            pr_title="BREAKING CHANGE: Remove old API",
        )
        assert features.has_breaking_keyword == 1.0

    def test_empty_diff(self) -> None:
        diff = ParsedDiff()
        features = extract_features(diff)
        assert features.num_files_changed == 0
        assert features.total_additions == 0

    def test_to_array(self) -> None:
        diff = _make_parsed_diff()
        features = extract_features(diff)
        arr = features.to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert len(arr) == len(features.feature_names)

    def test_to_dict(self) -> None:
        diff = _make_parsed_diff()
        features = extract_features(diff)
        d = features.to_dict()

        assert isinstance(d, dict)
        assert "num_files_changed" in d
        assert "total_additions" in d


class TestHistoricalFeatures:
    def test_enrich_with_bug_history(self) -> None:
        features = PRFeatures()
        enriched = enrich_with_historical_features(
            features,
            file_bug_counts={"main.py": 5, "utils.py": 1},
        )
        assert enriched.max_file_bug_history == 5.0
        assert enriched.avg_file_bug_history == 3.0

    def test_enrich_with_author_stats(self) -> None:
        features = PRFeatures()
        enriched = enrich_with_historical_features(
            features,
            author_stats={"total_commits": 200, "bug_rate": 0.12},
        )
        assert enriched.author_total_commits == 200.0
        assert enriched.author_bug_rate == 0.12
