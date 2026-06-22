"""
Celery application configuration.

Configures the Celery task queue for async PR analysis.
Workers process analysis jobs dispatched from the webhook endpoint.
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "pr_risk_analyzer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Concurrency
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)

    # Timeouts
    task_soft_time_limit=settings.celery_task_timeout,
    task_time_limit=settings.celery_task_timeout + 30,

    # Rate limiting
    task_default_rate_limit=f"{settings.max_concurrent_analyses}/m",

    # Auto-discover tasks
    include=["app.tasks.analyze_pr"],
)
