"""
Pydantic schemas for analysis endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FileRisk(BaseModel):
    """Risk assessment for a single file."""

    file_path: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    reasons: list[str] = Field(default_factory=list)


class RiskReport(BaseModel):
    """Complete risk report for a Pull Request."""

    risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score (0.0 - 1.0)")
    risk_level: RiskLevel
    risk_percentage: int = Field(ge=0, le=100, description="Risk as integer percentage")
    explanation: str = ""
    risky_files: list[FileRisk] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    model_version: str = ""
    analysis_duration_ms: int = 0


class AnalysisResponse(BaseModel):
    """Response for analysis endpoints."""

    status: str
    message: str
    analysis_id: UUID | None = None
    report: RiskReport | None = None


class AnalysisSummary(BaseModel):
    """Summary of a past analysis (for listing)."""

    analysis_id: UUID
    pr_number: int
    pr_title: str
    repo: str
    risk_score: float
    risk_level: RiskLevel
    created_at: datetime


class ManualAnalysisRequest(BaseModel):
    """Request body for manually triggering a PR analysis."""

    owner: str
    repo: str
    pr_number: int = Field(gt=0)
