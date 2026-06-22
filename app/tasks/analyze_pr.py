"""
PR Analysis Celery Task.

This is the main async task that:
1. Fetches PR data from GitHub
2. Parses the diff
3. Extracts features
4. Runs model inference
5. Generates a risk report
6. Posts a comment on the PR
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _run_async(coro):  # type: ignore[no-untyped-def]
    """Helper to run async code inside Celery (which runs sync)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an event loop — create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(
    bind=True,
    name="analyze_pr",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def analyze_pr_task(self: Any, event_data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze a Pull Request asynchronously.

    This task is dispatched from the webhook endpoint and runs
    in a Celery worker process.

    Args:
        event_data: Serialized WebhookEvent dict containing:
            - repo_owner, repo_name, pr_number, pr_title, etc.

    Returns:
        Dict with analysis results (risk_score, risk_level, etc.)
    """
    start_time = time.perf_counter()

    repo = f"{event_data['repo_owner']}/{event_data['repo_name']}"
    pr_number = event_data["pr_number"]

    logger.info("analysis_started", repo=repo, pr=pr_number, task_id=self.request.id)

    try:
        result = _run_async(_analyze_pr(event_data))

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "analysis_completed",
            repo=repo,
            pr=pr_number,
            risk_score=result.get("risk_score"),
            risk_level=result.get("risk_level"),
            duration_ms=duration_ms,
        )

        return result

    except Exception as exc:
        logger.error(
            "analysis_failed",
            repo=repo,
            pr=pr_number,
            error=str(exc),
            retry=self.request.retries,
        )
        raise self.retry(exc=exc)


async def _analyze_pr(event_data: dict[str, Any]) -> dict[str, Any]:
    """
    Core analysis pipeline (async).

    Steps:
    1. Fetch PR data from GitHub API
    2. Parse the diff
    3. Extract features
    4. Run model inference
    5. Build risk report
    6. Post comment on PR
    7. Store results in database
    """
    from app.core.diff_parser import create_model_input, parse_diff
    from app.core.feature_engine import extract_features
    from app.core.github_client import GitHubClient
    from app.ml.model import build_risk_report, classify_risk_level
    from app.ml.risk_report import format_risk_comment

    owner = event_data["repo_owner"]
    repo_name = event_data["repo_name"]
    pr_number = event_data["pr_number"]

    start = time.perf_counter()

    # --- Step 1: Fetch PR Data ---
    async with GitHubClient() as gh:
        pr_data = await gh.fetch_pr_data(owner, repo_name, pr_number)

        # --- Step 2: Parse Diff ---
        parsed_diff = parse_diff(pr_data.diff_text)

        # --- Step 3: Extract Features ---
        commit_messages = [c["message"] for c in pr_data.commits]
        features = extract_features(
            parsed_diff=parsed_diff,
            pr_title=pr_data.title,
            pr_description=pr_data.description,
            commit_messages=commit_messages,
        )

        # --- Step 4: Model Inference ---
        # TODO: Load model from registry and run inference
        # For now, use heuristic-based scoring from features
        from app.ml.model import Prediction

        heuristic_score = _heuristic_risk_score(features)
        prediction = Prediction(risk_score=heuristic_score)

        # --- Step 5: Build Risk Report ---
        duration_ms = int((time.perf_counter() - start) * 1000)
        report = build_risk_report(
            prediction=prediction,
            parsed_diff=parsed_diff,
            features=features,
            model_version="heuristic-v0.1.0",
            analysis_duration_ms=duration_ms,
        )

        # --- Step 6: Post Comment ---
        comment_body = format_risk_comment(report)
        comment_id = await gh.post_comment(owner, repo_name, pr_number, comment_body)

        # --- Step 7: Store in Database ---
        # TODO: Store PRAnalysis record

    return {
        "risk_score": report.risk_score,
        "risk_level": report.risk_level.value,
        "risk_percentage": report.risk_percentage,
        "num_risky_files": len(report.risky_files),
        "comment_id": comment_id,
        "duration_ms": duration_ms,
    }


def _heuristic_risk_score(features: Any) -> float:
    """
    Temporary heuristic risk scoring until a trained model is deployed.

    Based on empirical patterns from defect prediction research.
    """
    score = 0.0

    # File count (log-scaled)
    if features.num_files_changed > 20:
        score += 0.20
    elif features.num_files_changed > 10:
        score += 0.15
    elif features.num_files_changed > 5:
        score += 0.08

    # Change volume
    if features.total_lines_modified > 500:
        score += 0.20
    elif features.total_lines_modified > 200:
        score += 0.12
    elif features.total_lines_modified > 50:
        score += 0.05

    # No tests
    if features.has_test_changes == 0 and features.num_source_files > 0:
        score += 0.15

    # High entropy (scattered changes)
    if features.file_change_entropy > 3.0:
        score += 0.10
    elif features.file_change_entropy > 2.0:
        score += 0.05

    # Interleaved changes (complex modifications)
    if features.num_interleaved_changes > 5:
        score += 0.10
    elif features.num_interleaved_changes > 2:
        score += 0.05

    # Short description (poor documentation)
    if features.description_length < 20:
        score += 0.05

    # Historical bug rate (if available)
    if features.max_file_bug_history > 3:
        score += 0.15
    elif features.max_file_bug_history > 1:
        score += 0.08

    return min(score, 1.0)
