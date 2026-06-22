"""
Pydantic schemas for GitHub webhook events.
"""

from __future__ import annotations

from pydantic import BaseModel


class WebhookEvent(BaseModel):
    """Parsed data from a GitHub webhook pull_request event."""

    delivery_id: str
    event_type: str
    action: str
    pr_number: int
    pr_title: str
    pr_author: str
    repo_owner: str
    repo_name: str
    pr_url: str
    diff_url: str


class WebhookResponse(BaseModel):
    """Response sent back to GitHub after processing a webhook."""

    status: str  # "accepted", "ignored", "error"
    message: str
    pr_number: int | None = None
    repo: str | None = None
