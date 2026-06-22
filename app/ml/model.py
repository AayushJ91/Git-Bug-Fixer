"""
ML Model Loading and Inference.

Handles loading trained models (XGBoost baseline and CodeBERT)
and running inference on feature vectors / tokenized inputs.

Supports multiple model types via a unified interface:
- XGBoost (baseline)
- CodeBERT fine-tuned (primary)
- Hybrid (CodeBERT + handcrafted features)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.core.diff_parser import ParsedDiff
from app.core.feature_engine import PRFeatures, extract_features
from app.schemas.analysis import FileRisk, RiskLevel, RiskReport

logger = structlog.get_logger(__name__)


@dataclass
class Prediction:
    """Raw model prediction before post-processing."""

    risk_score: float  # 0.0 - 1.0
    feature_importances: dict[str, float] | None = None
    attention_weights: list[float] | None = None
    raw_logits: float | None = None


class BaseRiskModel(ABC):
    """Abstract base class for all risk prediction models."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load a trained model from disk."""
        ...

    @abstractmethod
    def predict(self, features: PRFeatures) -> Prediction:
        """Run inference on a single PR's features."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        ...


class XGBoostRiskModel(BaseRiskModel):
    """
    XGBoost-based risk prediction model.

    Uses handcrafted features only. Serves as the baseline
    model and the first model deployed in production.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._version: str = "xgboost-v0.1.0"
        self._feature_names: list[str] = []

    def load(self, model_path: str) -> None:
        """Load a trained XGBoost model from a JSON file."""
        import xgboost as xgb

        path = Path(model_path)
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path / "model.json"))

        # Load feature names
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self._version = meta.get("version", self._version)
                self._feature_names = meta.get("feature_names", [])

        logger.info("xgboost_model_loaded", version=self._version, path=model_path)

    def predict(self, features: PRFeatures) -> Prediction:
        """Predict risk score using handcrafted features."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        feature_array = features.to_array().reshape(1, -1)
        proba = self._model.predict_proba(feature_array)[0][1]  # Probability of class 1

        # Get feature importances for this prediction
        importances = self._model.feature_importances_
        feature_importance_dict = {}
        names = self._feature_names or features.feature_names
        for name, importance in zip(names, importances):
            feature_importance_dict[name] = float(importance)

        return Prediction(
            risk_score=float(proba),
            feature_importances=feature_importance_dict,
        )

    @property
    def model_version(self) -> str:
        return self._version


class CodeBERTRiskModel(BaseRiskModel):
    """
    CodeBERT-based risk prediction model.

    Fine-tuned microsoft/codebert-base with a classification head.
    Takes tokenized diff text as input.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._version: str = "codebert-v0.1.0"
        self._device: str = "cpu"
        self._max_length: int = 512

    def load(self, model_path: str) -> None:
        """Load fine-tuned CodeBERT model and tokenizer."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(path),
            num_labels=2,
        )
        self._model.eval()

        # Determine device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model = self._model.to(self._device)

        # Load metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self._version = meta.get("version", self._version)
                self._max_length = meta.get("max_length", self._max_length)

        logger.info(
            "codebert_model_loaded",
            version=self._version,
            device=self._device,
            path=model_path,
        )

    def predict(self, features: PRFeatures) -> Prediction:
        """
        Predict risk score using CodeBERT.

        Note: This method expects the model input text to be set
        externally. For the full pipeline, use predict_from_text().
        """
        raise NotImplementedError("Use predict_from_text() for CodeBERT model")

    def predict_from_text(
        self,
        text: str,
        return_attention: bool = False,
    ) -> Prediction:
        """
        Predict risk score from raw text input.

        Args:
            text: Formatted model input text (from create_model_input).
            return_attention: Whether to return attention weights for explainability.

        Returns:
            Prediction with risk score and optional attention weights.
        """
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize
        inputs = self._tokenizer(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # Inference
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_attentions=return_attention,
            )

        logits = outputs.logits[0]
        proba = torch.softmax(logits, dim=-1)[1].item()  # P(buggy)

        # Extract attention weights if requested
        attention = None
        if return_attention and outputs.attentions:
            # Average attention across all heads in the last layer
            last_layer_attention = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            avg_attention = last_layer_attention.mean(dim=0)  # (seq_len, seq_len)
            # CLS token attention to all other tokens
            cls_attention = avg_attention[0].cpu().numpy().tolist()
            attention = cls_attention

        return Prediction(
            risk_score=float(proba),
            attention_weights=attention,
            raw_logits=float(logits[1]),
        )

    @property
    def model_version(self) -> str:
        return self._version


# --- Risk Level Classification ---

def classify_risk_level(risk_score: float) -> RiskLevel:
    """
    Classify a risk score into Low/Medium/High.

    Thresholds are configurable. Defaults:
    - Low:    0.0 - 0.3
    - Medium: 0.3 - 0.7
    - High:   0.7 - 1.0
    """
    if risk_score >= 0.7:
        return RiskLevel.HIGH
    elif risk_score >= 0.3:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def build_risk_report(
    prediction: Prediction,
    parsed_diff: ParsedDiff,
    features: PRFeatures,
    model_version: str,
    analysis_duration_ms: int = 0,
) -> RiskReport:
    """
    Build a complete RiskReport from model prediction and features.

    Generates file-level risk breakdown and human-readable explanations.
    """
    risk_level = classify_risk_level(prediction.risk_score)
    risk_percentage = int(prediction.risk_score * 100)

    # --- File-Level Risk ---
    file_risks: list[FileRisk] = []
    for file_diff in parsed_diff.files:
        # Heuristic file-level risk (will be replaced by per-file model in v2)
        file_risk_score = _estimate_file_risk(file_diff, features)
        file_risks.append(
            FileRisk(
                file_path=file_diff.file_path,
                risk_score=file_risk_score,
                risk_level=classify_risk_level(file_risk_score),
                reasons=_get_file_risk_reasons(file_diff, features),
            )
        )

    # Sort by risk (highest first)
    file_risks.sort(key=lambda f: f.risk_score, reverse=True)

    # --- Explanations ---
    explanation = _generate_explanation(prediction, features, parsed_diff)
    recommendations = _generate_recommendations(features, parsed_diff)

    return RiskReport(
        risk_score=prediction.risk_score,
        risk_level=risk_level,
        risk_percentage=risk_percentage,
        explanation=explanation,
        risky_files=file_risks[:10],  # Top 10 riskiest files
        recommendations=recommendations,
        model_version=model_version,
        analysis_duration_ms=analysis_duration_ms,
    )


def _estimate_file_risk(file_diff: Any, features: PRFeatures) -> float:
    """Heuristic file-level risk estimation (placeholder for per-file model)."""
    risk = 0.0

    # Larger changes are riskier
    if file_diff.total_changes > 100:
        risk += 0.3
    elif file_diff.total_changes > 50:
        risk += 0.2
    elif file_diff.total_changes > 20:
        risk += 0.1

    # Interleaved additions/deletions are riskier
    interleaved_hunks = sum(
        1 for h in file_diff.hunks if h.added_lines and h.deleted_lines
    )
    risk += min(interleaved_hunks * 0.1, 0.3)

    # More hunks = more scattered changes = riskier
    if len(file_diff.hunks) > 5:
        risk += 0.2
    elif len(file_diff.hunks) > 2:
        risk += 0.1

    return min(risk, 1.0)


def _get_file_risk_reasons(file_diff: Any, features: PRFeatures) -> list[str]:
    """Generate human-readable reasons for file-level risk."""
    reasons = []

    if file_diff.total_changes > 100:
        reasons.append(f"Large change: {file_diff.total_changes} lines modified")

    interleaved = sum(
        1 for h in file_diff.hunks if h.added_lines and h.deleted_lines
    )
    if interleaved > 0:
        reasons.append(f"{interleaved} hunk(s) with interleaved additions and deletions")

    if len(file_diff.hunks) > 5:
        reasons.append(f"Scattered changes across {len(file_diff.hunks)} hunks")

    if file_diff.change_type.value == "delete":
        reasons.append("File deletion — verify no remaining references")

    return reasons


def _generate_explanation(
    prediction: Prediction,
    features: PRFeatures,
    parsed_diff: ParsedDiff,
) -> str:
    """Generate a human-readable explanation of the overall risk."""
    parts = []

    if features.num_files_changed > 10:
        parts.append(
            f"This PR modifies {int(features.num_files_changed)} files, "
            "which is above the typical threshold. Large PRs are harder to review."
        )

    if features.total_lines_modified > 500:
        parts.append(
            f"Total of {int(features.total_lines_modified)} lines changed. "
            "Large changes carry higher risk of introducing bugs."
        )

    if features.has_test_changes == 0 and features.num_source_files > 0:
        parts.append(
            "No test files were modified. Consider adding tests for changed code paths."
        )

    if features.file_change_entropy > 3.0:
        parts.append(
            "Changes are spread across many files/directories (high entropy), "
            "suggesting a cross-cutting change that may have unintended side effects."
        )

    if features.num_interleaved_changes > 3:
        parts.append(
            f"{int(features.num_interleaved_changes)} hunks have interleaved "
            "additions and deletions, indicating complex refactoring or logic changes."
        )

    if features.max_file_bug_history > 2:
        parts.append(
            f"Files with {int(features.max_file_bug_history)} prior bug fixes are being modified. "
            "Historical bug frequency is a strong predictor of future bugs."
        )

    if not parts:
        if prediction.risk_score < 0.3:
            parts.append("This PR appears to be a low-risk change with a manageable scope.")
        else:
            parts.append("This PR has been flagged for moderate-to-high risk based on change patterns.")

    return " ".join(parts)


def _generate_recommendations(features: PRFeatures, parsed_diff: ParsedDiff) -> list[str]:
    """Generate actionable recommendations based on risk signals."""
    recs = []

    if features.has_test_changes == 0 and features.num_source_files > 0:
        recs.append("Add unit tests for the changed code paths.")

    if features.num_files_changed > 15:
        recs.append("Consider splitting this PR into smaller, focused changes.")

    if features.description_length < 50:
        recs.append("Add a detailed PR description to help reviewers understand the changes.")

    if features.has_issue_reference == 0:
        recs.append("Link this PR to a related issue for better traceability.")

    if features.num_interleaved_changes > 5:
        recs.append("Many interleaved changes detected. Consider separating refactoring from logic changes.")

    return recs
