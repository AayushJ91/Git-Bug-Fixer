"""
Dashboard API endpoints.

Provides aggregate statistics and metrics for monitoring
the system's performance and usage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/stats")
async def get_dashboard_stats() -> dict[str, Any]:
    """
    Aggregate statistics for the dashboard.

    Returns:
    - Total PRs analyzed
    - Risk distribution (low/medium/high)
    - Average analysis time
    - Model performance metrics
    """
    # TODO: Query database for aggregate stats
    return {
        "total_analyses": 0,
        "risk_distribution": {"low": 0, "medium": 0, "high": 0},
        "avg_analysis_duration_ms": 0,
        "model_version": "0.1.0",
        "last_analysis_at": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/model/info")
async def get_model_info() -> dict[str, Any]:
    """
    Information about the currently active ML model.

    Returns model version, type, metrics, and deployment timestamp.
    """
    # TODO: Query model registry
    return {
        "model_name": "xgboost-baseline-v1",
        "model_type": "xgboost",
        "is_active": True,
        "metrics": {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1_score": 0.0,
            "mcc": 0.0,
        },
        "deployed_at": None,
    }
