"""
CodeBERT Fine-Tuning Script.

Fine-tunes microsoft/codebert-base for binary classification
(safe vs. bug-inducing) on tokenized diff data.

Implements:
- Phased training (frozen → unfrozen)
- Mixed precision (fp16)
- Early stopping
- MLflow tracking
- Gradient accumulation

Usage:
    python -m ml_pipeline.training.train_codebert \
        --data-path data/processed/tokenized_diffs.parquet \
        --output-dir models/codebert-v1 \
        --epochs 10 \
        --batch-size 16
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = structlog.get_logger(__name__)


class DiffDataset(Dataset):
    """
    PyTorch Dataset for tokenized diff data.

    Each sample contains:
    - input_ids: Tokenized diff text
    - attention_mask: Attention mask
    - label: 0 (safe) or 1 (buggy)
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_codebert(
    data_path: str,
    output_dir: str,
    model_name: str = "microsoft/codebert-base",
    max_length: int = 512,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    freeze_epochs: int = 2,
    patience: int = 3,
    gradient_accumulation_steps: int = 1,
    experiment_name: str = "pr-risk-codebert",
) -> dict[str, float]:
    """
    Fine-tune CodeBERT on bug-inducing change detection.

    Training phases:
    1. Frozen: Train only classification head (freeze_epochs)
    2. Unfrozen: Fine-tune all layers with small learning rate

    Args:
        data_path: Path to processed dataset (Parquet with 'text' and 'label' columns)
        output_dir: Directory to save the fine-tuned model
        model_name: HuggingFace model name
        max_length: Maximum token sequence length
        epochs: Total training epochs
        batch_size: Training batch size
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup steps
        weight_decay: AdamW weight decay
        freeze_epochs: Number of epochs to freeze encoder
        patience: Early stopping patience
        gradient_accumulation_steps: Accumulate gradients over N steps

    Returns:
        Best validation metrics dict
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    # --- Setup ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("training_setup", device=str(device), model=model_name)

    # --- Load Data ---
    df = pd.read_parquet(data_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Split (use repo_id for group split if available)
    if "repo_id" in df.columns:
        # Group-aware split
        unique_repos = df["repo_id"].unique()
        train_repos, val_repos = train_test_split(
            unique_repos, test_size=0.15, random_state=42
        )
        train_mask = df["repo_id"].isin(train_repos)
        val_mask = df["repo_id"].isin(val_repos)
        train_texts = df[train_mask]["text"].tolist()
        train_labels = df[train_mask]["label"].tolist()
        val_texts = df[val_mask]["text"].tolist()
        val_labels = df[val_mask]["label"].tolist()
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=42
        )

    logger.info(
        "data_split",
        train=len(train_texts),
        val=len(val_texts),
        train_positive_rate=f"{np.mean(train_labels) * 100:.1f}%",
    )

    # --- Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to(device)

    # --- Datasets ---
    train_dataset = DiffDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = DiffDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set > 0 on Linux
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --- Class weights for imbalanced data ---
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    class_weights = torch.tensor(
        [1.0, neg_count / max(pos_count, 1)], dtype=torch.float32
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Mixed Precision ---
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # --- MLflow ---
    mlflow.set_experiment(experiment_name)

    best_val_auc = 0.0
    patience_counter = 0
    best_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name=f"codebert-{int(time.time())}"):
        mlflow.log_params({
            "model_name": model_name,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "freeze_epochs": freeze_epochs,
            "class_weight_positive": float(class_weights[1]),
        })

        for epoch in range(epochs):
            # --- Phase 1: Freeze encoder ---
            if epoch < freeze_epochs:
                for param in model.roberta.parameters():
                    param.requires_grad = False
                logger.info("encoder_frozen", epoch=epoch)
            elif epoch == freeze_epochs:
                for param in model.roberta.parameters():
                    param.requires_grad = True
                # Reduce learning rate for unfrozen training
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate / 5
                logger.info("encoder_unfrozen", epoch=epoch)

            # --- Training ---
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_tensor = batch["label"].to(device)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, labels_tensor)
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels_tensor)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

            avg_train_loss = total_loss / num_batches

            # --- Validation ---
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(batch["label"].numpy())

            all_preds_np = np.array(all_preds)
            all_labels_np = np.array(all_labels)

            val_auc_roc = roc_auc_score(all_labels_np, all_preds_np)
            val_auc_pr = average_precision_score(all_labels_np, all_preds_np)
            val_f1 = f1_score(all_labels_np, (all_preds_np > 0.5).astype(int))

            epoch_metrics = {
                "train_loss": avg_train_loss,
                "val_auc_roc": val_auc_roc,
                "val_auc_pr": val_auc_pr,
                "val_f1": val_f1,
            }

            mlflow.log_metrics(epoch_metrics, step=epoch)

            logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                **{k: f"{v:.4f}" for k, v in epoch_metrics.items()},
            )

            # --- Early Stopping ---
            if val_auc_roc > best_val_auc:
                best_val_auc = val_auc_roc
                best_metrics = epoch_metrics
                patience_counter = 0

                # Save best model
                save_path = Path(output_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(save_path))
                tokenizer.save_pretrained(str(save_path))

                # Save metadata
                metadata = {
                    "version": "codebert-v0.1.0",
                    "model_name": model_name,
                    "max_length": max_length,
                    "metrics": {k: float(v) for k, v in best_metrics.items()},
                    "epoch": epoch + 1,
                }
                with open(save_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info("best_model_saved", auc_roc=f"{best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("early_stopping", epoch=epoch + 1, best_auc=f"{best_val_auc:.4f}")
                    break

        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

    logger.info("training_complete", best_metrics=best_metrics)
    return best_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT")
    parser.add_argument("--data-path", required=True, help="Path to tokenized dataset")
    parser.add_argument("--output-dir", default="models/codebert-v1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)

    args = parser.parse_args()

    train_codebert(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
