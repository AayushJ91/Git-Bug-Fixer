"""
Tests for the webhook endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _make_signature(payload: bytes, secret: str = "") -> str:
    """Generate a valid HMAC-SHA256 signature for a payload."""
    digest = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


SAMPLE_PR_PAYLOAD = {
    "action": "opened",
    "pull_request": {
        "number": 42,
        "title": "Fix database connection pooling",
        "body": "This PR fixes connection pooling issues.",
        "user": {"login": "testuser"},
        "html_url": "https://github.com/owner/repo/pull/42",
        "diff_url": "https://github.com/owner/repo/pull/42.diff",
    },
    "repository": {
        "name": "repo",
        "owner": {"login": "owner"},
    },
}


class TestWebhook:
    def test_health_check(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "version" in data

    def test_webhook_pr_opened(self) -> None:
        payload = json.dumps(SAMPLE_PR_PAYLOAD).encode()
        response = client.post(
            "/webhook/",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "test-delivery-123",
                "X-Hub-Signature-256": "",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["pr_number"] == 42

    def test_webhook_non_pr_event(self) -> None:
        response = client.post(
            "/webhook/",
            json={"action": "created"},
            headers={
                "X-GitHub-Event": "issue_comment",
                "X-GitHub-Delivery": "test-delivery-456",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ignored"

    def test_webhook_irrelevant_pr_action(self) -> None:
        payload = {**SAMPLE_PR_PAYLOAD, "action": "closed"}
        response = client.post(
            "/webhook/",
            json=payload,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "test-delivery-789",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"
