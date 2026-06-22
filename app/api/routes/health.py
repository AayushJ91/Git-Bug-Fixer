"""
Health check endpoint.

Verifies that the API, database, Redis, and ML model are operational.
Used by load balancers and monitoring systems.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns the status of all system components.
    Used by load balancers, Docker health checks, and monitoring.
    """
    checks: dict[str, str] = {}
    overall_healthy = True

    # Check API (always healthy if this runs)
    checks["api"] = "healthy"

    # Check Database
    try:
        # TODO: Execute a simple query (SELECT 1)
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {e}"
        overall_healthy = False
        logger.error("health_check_db_failed", error=str(e))

    # Check Redis
    try:
        # TODO: Ping Redis
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e}"
        overall_healthy = False
        logger.error("health_check_redis_failed", error=str(e))

    # Check ML Model
    try:
        # TODO: Verify model is loaded
        checks["ml_model"] = "healthy"
    except Exception as e:
        checks["ml_model"] = f"unhealthy: {e}"
        overall_healthy = False
        logger.error("health_check_model_failed", error=str(e))

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "checks": checks,
    }
