"""
Baseline Model Training Script.

Trains Logistic Regression, Random Forest, and XGBoost models
on handcrafted features extracted from commit/PR data.

This establishes a performance baseline before fine-tuning
transformer models.

Usage:
    python -m ml_pipeline.training.train_baseline \
        --data-path data/processed/features.parquet \
        --output-dir models/baseline
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load the training dataset from a Parquet file.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) — 0=safe, 1=buggy
        groups: Repository IDs for group-aware splitting
        feature_names: Ordered list of feature names
    """
    import pandas as pd

    df = pd.read_parquet(data_path)

    # Separate features, labels, and groups
    label_col = "label"
    group_col = "repo_id"

    feature_cols = [
        col for col in df.columns
        if col not in [label_col, group_col, "commit_hash", "file_path"]
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)
    groups = df[group_col].values if group_col in df.columns else np.zeros(len(df))

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "dataset_loaded",
        samples=len(y),
        features=len(feature_cols),
        positive_rate=f"{y.mean() * 100:.1f}%",
    )

    return X, y, groups, feature_cols


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Compute all evaluation metrics for a model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y, y_proba),
        "auc_pr": average_precision_score(y, y_proba),
        "f1_score": f1_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
        "precision": float(classification_report(y, y_pred, output_dict=True)["1"]["precision"]),
        "recall": float(classification_report(y, y_pred, output_dict=True)["1"]["recall"]),
    }

    logger.info(f"{model_name}_metrics", **{k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def train_baselines(
    data_path: str,
    output_dir: str,
    n_folds: int = 5,
    experiment_name: str = "pr-risk-baseline",
) -> dict[str, dict[str, float]]:
    """
    Train and evaluate baseline models with repository-level cross-validation.

    Models trained:
    1. Logistic Regression (sanity check)
    2. Random Forest
    3. XGBoost

    All metrics are logged to MLflow.
    """
    import xgboost as xgb

    X, y, groups, feature_names = load_dataset(data_path)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Models ---
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=float(np.sum(y == 0) / max(np.sum(y == 1), 1)),
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict[str, float]] = {}

    # Setup MLflow
    mlflow.set_experiment(experiment_name)

    # --- Cross-Validation ---
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for model_name, model in models.items():
        logger.info(f"training_{model_name}")

        fold_metrics: list[dict[str, float]] = []

        with mlflow.start_run(run_name=model_name):
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y, groups)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_val, y_val, f"{model_name}_fold{fold_idx}")
                fold_metrics.append(metrics)

            # Average metrics across folds
            avg_metrics = {
                key: float(np.mean([m[key] for m in fold_metrics]))
                for key in fold_metrics[0]
            }
            std_metrics = {
                f"{key}_std": float(np.std([m[key] for m in fold_metrics]))
                for key in fold_metrics[0]
            }

            # Log to MLflow
            mlflow.log_params({"model_type": model_name, "n_folds": n_folds})
            mlflow.log_metrics(avg_metrics)
            mlflow.log_metrics(std_metrics)

            results[model_name] = avg_metrics

            logger.info(
                f"{model_name}_cv_results",
                **{k: f"{v:.4f}" for k, v in avg_metrics.items()},
            )

    # --- Train final best model on all data ---
    best_model_name = max(results, key=lambda k: results[k]["auc_roc"])
    logger.info("best_model", name=best_model_name, auc_roc=results[best_model_name]["auc_roc"])

    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)

    # Save model
    output_path = Path(output_dir) / best_model_name
    output_path.mkdir(parents=True, exist_ok=True)

    if best_model_name == "xgboost":
        best_model.save_model(str(output_path / "model.json"))
    else:
        import joblib
        joblib.dump(best_model, str(output_path / "model.joblib"))

    # Save scaler
    import joblib
    joblib.dump(scaler, str(output_path / "scaler.joblib"))

    # Save metadata
    metadata = {
        "version": f"{best_model_name}-v0.1.0",
        "model_type": best_model_name,
        "feature_names": feature_names,
        "metrics": results[best_model_name],
        "n_training_samples": len(y),
        "positive_rate": float(y.mean()),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("model_saved", path=str(output_path))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--data-path", required=True, help="Path to processed features Parquet")
    parser.add_argument("--output-dir", default="models/baseline", help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    results = train_baselines(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )

    print("\n=== Results ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
