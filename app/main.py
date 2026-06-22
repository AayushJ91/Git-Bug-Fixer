"""
FastAPI application factory.

Creates and configures the FastAPI app with:
- CORS middleware
- Request logging
- Startup/shutdown lifecycle hooks
- API route registration
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analysis, dashboard, health, webhook
from app.config import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.app_env.value,
        debug=settings.debug,
    )

    # --- Startup ---
    # Initialize ML model (lazy-loaded on first request if not preloaded)
    # Initialize database connection pool
    # Verify GitHub token validity
    logger.info("startup_complete")

    yield

    # --- Shutdown ---
    logger.info("application_shutting_down")
    # Close database connections
    # Flush metrics
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title="AI Pull Request Risk Analyzer",
        description=(
            "Automatically analyzes GitHub Pull Requests and predicts "
            "the likelihood of bug introduction using transformer-based models."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Log every request with timing information."""
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    # --- Routes ---
    app.include_router(health.router, tags=["Health"])
    app.include_router(webhook.router, prefix="/webhook", tags=["Webhook"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

    return app


# Application instance for uvicorn
app = create_app()
