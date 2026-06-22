"""
GitHub API client.

Handles all interactions with the GitHub REST API:
- Fetching PR metadata (title, description, author)
- Fetching PR diffs (unified diff format)
- Fetching commit information
- Posting analysis comments on PRs

Uses httpx for async HTTP and tenacity for retry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = structlog.get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"


@dataclass
class PRData:
    """Structured representation of a Pull Request."""

    owner: str
    repo: str
    number: int
    title: str = ""
    description: str = ""
    author: str = ""
    head_sha: str = ""
    base_branch: str = ""
    head_branch: str = ""
    created_at: str = ""
    diff_text: str = ""
    commits: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[dict[str, Any]] = field(default_factory=list)
    num_additions: int = 0
    num_deletions: int = 0
    num_changed_files: int = 0


class GitHubClient:
    """
    Async GitHub API client.

    Usage:
        async with GitHubClient() as client:
            pr_data = await client.fetch_pr_data("owner", "repo", 42)
            await client.post_comment("owner", "repo", 42, "Risk: HIGH")
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or get_settings().github_token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GitHubClient:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            headers=headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET request with retry logic."""
        response = await self.client.get(url, **kwargs)
        response.raise_for_status()
        return response

    async def fetch_pr_data(self, owner: str, repo: str, pr_number: int) -> PRData:
        """
        Fetch complete PR data including metadata, diff, and commits.

        Makes 3 API calls:
        1. PR metadata (title, description, author, etc.)
        2. PR diff (unified diff text)
        3. PR files (list of changed files with patches)
        """
        logger.info("fetching_pr_data", repo=f"{owner}/{repo}", pr=pr_number)

        # 1. PR Metadata
        pr_resp = await self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        pr_json = pr_resp.json()

        # 2. PR Diff
        diff_resp = await self.client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        diff_text = diff_resp.text if diff_resp.status_code == 200 else ""

        # 3. PR Files
        files_resp = await self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        files_json = files_resp.json()

        # 4. PR Commits
        commits_resp = await self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}/commits")
        commits_json = commits_resp.json()

        pr_data = PRData(
            owner=owner,
            repo=repo,
            number=pr_number,
            title=pr_json.get("title", ""),
            description=pr_json.get("body", "") or "",
            author=pr_json.get("user", {}).get("login", ""),
            head_sha=pr_json.get("head", {}).get("sha", ""),
            base_branch=pr_json.get("base", {}).get("ref", ""),
            head_branch=pr_json.get("head", {}).get("ref", ""),
            created_at=pr_json.get("created_at", ""),
            diff_text=diff_text,
            commits=[
                {
                    "sha": c.get("sha", ""),
                    "message": c.get("commit", {}).get("message", ""),
                    "author": c.get("commit", {}).get("author", {}).get("name", ""),
                }
                for c in commits_json
            ],
            files_changed=[
                {
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "changes": f.get("changes", 0),
                    "patch": f.get("patch", ""),
                }
                for f in files_json
            ],
            num_additions=pr_json.get("additions", 0),
            num_deletions=pr_json.get("deletions", 0),
            num_changed_files=pr_json.get("changed_files", 0),
        )

        logger.info(
            "pr_data_fetched",
            repo=f"{owner}/{repo}",
            pr=pr_number,
            files=pr_data.num_changed_files,
            additions=pr_data.num_additions,
            deletions=pr_data.num_deletions,
        )

        return pr_data

    async def post_comment(self, owner: str, repo: str, pr_number: int, body: str) -> int:
        """
        Post an analysis comment on a PR.

        Returns the comment ID for future reference (e.g., updating the comment).
        """
        response = await self.client.post(
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        response.raise_for_status()
        comment_id = response.json().get("id", 0)

        logger.info(
            "comment_posted",
            repo=f"{owner}/{repo}",
            pr=pr_number,
            comment_id=comment_id,
        )
        return comment_id

    async def update_comment(self, owner: str, repo: str, comment_id: int, body: str) -> None:
        """Update an existing comment on a PR."""
        response = await self.client.patch(
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}",
            json={"body": body},
        )
        response.raise_for_status()
        logger.info("comment_updated", repo=f"{owner}/{repo}", comment_id=comment_id)

    async def get_rate_limit(self) -> dict[str, Any]:
        """Check current GitHub API rate limit status."""
        response = await self._get("/rate_limit")
        data = response.json()
        core = data.get("resources", {}).get("core", {})
        logger.info(
            "rate_limit",
            remaining=core.get("remaining"),
            limit=core.get("limit"),
            reset=core.get("reset"),
        )
        return core
