"""
Analysis API endpoints.

Provides endpoints for:
- Retrieving analysis results for a specific PR
- Listing analyses for a repository
- Manually triggering a PR analysis
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisSummary,
    ManualAnalysisRequest,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: UUID) -> Any:
    """
    Retrieve a specific PR analysis result by its ID.

    Returns the full risk report including score, level,
    risky files, and explanations.
    """
    # TODO: Query database for analysis result
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Analysis {analysis_id} not found",
    )


@router.get("/analysis/repo/{owner}/{repo}", response_model=list[AnalysisSummary])
async def list_repo_analyses(
    owner: str,
    repo: str,
    limit: int = Query(default=20, le=100, ge=1),
    offset: int = Query(default=0, ge=0),
) -> Any:
    """
    List all analyses for a given repository.

    Results are ordered by creation date (most recent first).
    Supports pagination via limit/offset.
    """
    logger.info("list_analyses", owner=owner, repo=repo, limit=limit, offset=offset)

    # TODO: Query database
    return []


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_analysis(request: ManualAnalysisRequest) -> Any:
    """
    Manually trigger an analysis for a specific Pull Request.

    Useful for re-analyzing a PR after model updates or
    for analyzing PRs from before the webhook was set up.
    """
    logger.info(
        "manual_analysis_triggered",
        repo=f"{request.owner}/{request.repo}",
        pr_number=request.pr_number,
    )

    # TODO: Dispatch to Celery task queue
    # from app.tasks.analyze_pr import analyze_pr_task
    # task = analyze_pr_task.delay(...)

    return AnalysisResponse(
        status="queued",
        message=f"Analysis queued for PR #{request.pr_number}",
    )
