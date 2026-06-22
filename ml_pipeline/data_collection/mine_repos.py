"""
Repository Mining Script.

Mines open-source repositories to extract commit data
for training the bug-inducing change classifier.

Uses PyDriller to efficiently traverse git history and
extract structured commit data.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Keywords that indicate a bug-fixing commit
BUG_FIX_KEYWORDS = [
    "fix", "bug", "defect", "patch", "resolve", "issue",
    "error", "fault", "fail", "crash", "broken", "incorrect",
    "wrong", "repair",
]

# Keywords that indicate a FALSE POSITIVE (not actually bug-fixing)
FALSE_POSITIVE_KEYWORDS = [
    "fix typo", "fix format", "fix style", "fix lint",
    "fix whitespace", "fix indent", "fix comment",
    "fix readme", "fix doc", "fix ci", "fix build config",
    "fix merge", "fix conflict",
]

# File extensions to include
SOURCE_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".rb", ".php", ".c", ".cpp", ".h",
    ".cs", ".swift", ".kt", ".scala",
}


@dataclass
class MinedCommit:
    """Structured commit data extracted from a repository."""

    repo_owner: str
    repo_name: str
    commit_hash: str
    parent_hash: str
    author_name: str
    author_email: str
    author_date: str
    commit_message: str
    is_merge: bool
    is_bug_fix: bool

    # File-level data
    files: list[dict[str, Any]] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    num_files_changed: int = 0


def is_bug_fix_commit(message: str) -> bool:
    """
    Determine if a commit message indicates a bug fix.

    Uses keyword matching with false positive filtering.
    """
    message_lower = message.lower()

    # Check for false positives first
    for fp in FALSE_POSITIVE_KEYWORDS:
        if fp in message_lower:
            return False

    # Check for bug-fix keywords
    for keyword in BUG_FIX_KEYWORDS:
        if keyword in message_lower:
            return True

    return False


def should_include_file(file_path: str) -> bool:
    """Check if a file should be included based on its extension."""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SOURCE_EXTENSIONS


def mine_repository(
    repo_url: str,
    output_dir: str,
    repo_owner: str = "",
    repo_name: str = "",
    max_commits: int = 5000,
    since: datetime | None = None,
    to: datetime | None = None,
) -> list[MinedCommit]:
    """
    Mine a single repository for commit data.

    Args:
        repo_url: Git URL or local path to the repository.
        output_dir: Directory to store extracted data.
        repo_owner: GitHub repository owner.
        repo_name: GitHub repository name.
        max_commits: Maximum number of commits to process.
        since: Only include commits after this date.
        to: Only include commits before this date.

    Returns:
        List of MinedCommit objects.
    """
    from pydriller import Repository

    logger.info(
        "mining_repository",
        repo=repo_url,
        max_commits=max_commits,
    )

    mined_commits: list[MinedCommit] = []
    processed = 0
    bug_fixes = 0

    repo_kwargs: dict[str, Any] = {"path_to_repo": repo_url}
    if since:
        repo_kwargs["since"] = since
    if to:
        repo_kwargs["to"] = to

    for commit in Repository(**repo_kwargs).traverse_commits():
        if processed >= max_commits:
            break

        # Skip merge commits
        if commit.merge:
            continue

        # Skip if no parents (initial commit)
        if len(commit.parents) == 0:
            continue

        is_fix = is_bug_fix_commit(commit.msg)
        if is_fix:
            bug_fixes += 1

        # Extract file-level data
        file_data = []
        total_add = 0
        total_del = 0

        for mod in commit.modified_files:
            # Skip non-source files
            file_path = mod.new_path or mod.old_path or ""
            if not should_include_file(file_path):
                continue

            file_info = {
                "filename": file_path,
                "old_path": mod.old_path or "",
                "new_path": mod.new_path or "",
                "change_type": mod.change_type.name if mod.change_type else "UNKNOWN",
                "additions": mod.added_lines,
                "deletions": mod.deleted_lines,
                "diff": mod.diff or "",
                "complexity": mod.complexity or 0,
                "nloc": mod.nloc or 0,
            }
            file_data.append(file_info)
            total_add += mod.added_lines
            total_del += mod.deleted_lines

        if not file_data:
            continue  # Skip commits with no relevant source files

        mined_commit = MinedCommit(
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_hash=commit.hash,
            parent_hash=commit.parents[0] if commit.parents else "",
            author_name=commit.author.name,
            author_email=commit.author.email,
            author_date=commit.author_date.isoformat(),
            commit_message=commit.msg[:1000],  # Truncate very long messages
            is_merge=commit.merge,
            is_bug_fix=is_fix,
            files=file_data,
            total_additions=total_add,
            total_deletions=total_del,
            num_files_changed=len(file_data),
        )

        mined_commits.append(mined_commit)
        processed += 1

        if processed % 500 == 0:
            logger.info(
                "mining_progress",
                processed=processed,
                bug_fixes=bug_fixes,
            )

    logger.info(
        "mining_complete",
        repo=repo_url,
        total_commits=processed,
        bug_fixes=bug_fixes,
        bug_fix_rate=f"{bug_fixes / max(processed, 1) * 100:.1f}%",
    )

    # Save to disk
    output_path = Path(output_dir) / f"{repo_owner}_{repo_name}_commits.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for commit in mined_commits:
            f.write(json.dumps(asdict(commit)) + "\n")

    logger.info("data_saved", path=str(output_path), records=len(mined_commits))

    return mined_commits


# --- Example target repositories ---
RECOMMENDED_REPOS = [
    # Python
    {"url": "https://github.com/django/django", "owner": "django", "name": "django"},
    {"url": "https://github.com/pallets/flask", "owner": "pallets", "name": "flask"},
    {"url": "https://github.com/psf/requests", "owner": "psf", "name": "requests"},
    {"url": "https://github.com/pandas-dev/pandas", "owner": "pandas-dev", "name": "pandas"},
    {"url": "https://github.com/scikit-learn/scikit-learn", "owner": "scikit-learn", "name": "scikit-learn"},
    # JavaScript
    {"url": "https://github.com/expressjs/express", "owner": "expressjs", "name": "express"},
    {"url": "https://github.com/lodash/lodash", "owner": "lodash", "name": "lodash"},
    # Java
    {"url": "https://github.com/spring-projects/spring-boot", "owner": "spring-projects", "name": "spring-boot"},
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine a GitHub repository for commit data")
    parser.add_argument("--repo-url", required=True, help="Git repository URL or local path")
    parser.add_argument("--owner", default="", help="Repository owner")
    parser.add_argument("--name", default="", help="Repository name")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--max-commits", type=int, default=5000, help="Max commits to process")

    args = parser.parse_args()

    mine_repository(
        repo_url=args.repo_url,
        output_dir=args.output_dir,
        repo_owner=args.owner,
        repo_name=args.name,
        max_commits=args.max_commits,
    )
