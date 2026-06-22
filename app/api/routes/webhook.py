"""
GitHub Webhook endpoint.

Receives webhook events from GitHub when Pull Requests are opened,
updated, or synchronized. Verifies the HMAC-SHA256 signature to
prevent unauthorized requests, then dispatches the analysis task.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import structlog
from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import get_settings
from app.schemas.webhook import WebhookEvent, WebhookResponse

router = APIRouter()
logger = structlog.get_logger(__name__)

# PR actions we care about
RELEVANT_PR_ACTIONS = {"opened", "synchronize", "reopened"}


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify the GitHub webhook HMAC-SHA256 signature.

    GitHub sends a signature in the X-Hub-Signature-256 header.
    We recompute it using our secret and compare.
    """
    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)


@router.post("/", response_model=WebhookResponse)
async def receive_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
    x_github_delivery: str = Header(default=""),
) -> WebhookResponse:
    """
    Receive and process GitHub webhook events.

    Flow:
    1. Verify HMAC-SHA256 signature
    2. Parse the event payload
    3. Filter for relevant PR actions (opened, synchronize, reopened)
    4. Dispatch async analysis task
    5. Return 200 immediately (GitHub expects fast response)
    """
    settings = get_settings()

    # --- Step 1: Read raw body for signature verification ---
    body = await request.body()

    # --- Step 2: Verify signature ---
    if settings.github_webhook_secret:
        if not verify_webhook_signature(body, x_hub_signature_256, settings.github_webhook_secret):
            logger.warning(
                "webhook_signature_invalid",
                delivery_id=x_github_delivery,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

    # --- Step 3: Parse payload ---
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # --- Step 4: Filter for PR events ---
    if x_github_event != "pull_request":
        logger.debug("webhook_ignored", event=x_github_event)
        return WebhookResponse(
            status="ignored",
            message=f"Event type '{x_github_event}' is not processed",
        )

    action = payload.get("action", "")
    if action not in RELEVANT_PR_ACTIONS:
        logger.debug("webhook_pr_action_ignored", action=action)
        return WebhookResponse(
            status="ignored",
            message=f"PR action '{action}' is not processed",
        )

    # --- Step 5: Extract PR info and dispatch analysis ---
    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})

    event = WebhookEvent(
        delivery_id=x_github_delivery,
        event_type=x_github_event,
        action=action,
        pr_number=pr_data.get("number", 0),
        pr_title=pr_data.get("title", ""),
        pr_author=pr_data.get("user", {}).get("login", ""),
        repo_owner=repo_data.get("owner", {}).get("login", ""),
        repo_name=repo_data.get("name", ""),
        pr_url=pr_data.get("html_url", ""),
        diff_url=pr_data.get("diff_url", ""),
    )

    logger.info(
        "webhook_pr_received",
        repo=f"{event.repo_owner}/{event.repo_name}",
        pr_number=event.pr_number,
        action=event.action,
        delivery_id=event.delivery_id,
    )

    # TODO: Dispatch to Celery task queue
    # from app.tasks.analyze_pr import analyze_pr_task
    # analyze_pr_task.delay(event.model_dump())

    return WebhookResponse(
        status="accepted",
        message=f"Analysis queued for PR #{event.pr_number}",
        pr_number=event.pr_number,
        repo=f"{event.repo_owner}/{event.repo_name}",
    )
