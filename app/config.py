"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
All settings are validated at startup — the app fails fast if
required configuration is missing.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "ai-pr-risk-analyzer"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/pranalyzer"
    database_url_sync: str = "postgresql+psycopg2://user:password@localhost:5432/pranalyzer"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- GitHub ---
    github_token: str = ""
    github_webhook_secret: str = ""
    github_app_id: str = ""
    github_app_private_key_path: str = ""

    # --- ML Model ---
    model_path: str = "./models/codebert-v1"
    model_name: str = "microsoft/codebert-base"
    max_sequence_length: int = 512
    inference_device: str = "cpu"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "pr-risk-analyzer"

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_timeout: int = 120

    # --- Rate Limiting ---
    max_concurrent_analyses: int = 10
    github_api_rate_limit: int = 5000

    # --- Security ---
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = Field(default_factory=lambda: ["dev-key-change-me"])

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Call this instead of constructing Settings()."""
    return Settings()
