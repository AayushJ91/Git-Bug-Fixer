# AI Pull Request Risk Analyzer 🤖

> Automatically analyze GitHub Pull Requests and predict the likelihood of bug introduction using transformer-based models.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![CodeBERT](https://img.shields.io/badge/model-CodeBERT-orange.svg)](https://huggingface.co/microsoft/codebert-base)

## Overview
hi
This system receives GitHub webhook events when Pull Requests are opened, analyzes the code changes using ML models (XGBoost baseline + CodeBERT transformer), and posts an automated risk assessment comment directly on the PR.

### What It Analyzes
- **PR metadata** — title, description, author
- **Commit messages** — intent and scope
- **Code diffs** — structural and semantic changes
- **File history** — past bug frequency (when available)

### What It Predicts
- **Risk Score** (0–100%) — calibrated probability of bug introduction
- **Risk Level** (Low / Medium / High) — actionable classification
- **File-Level Risk** — which specific files are riskiest and why
- **Recommendations** — actionable suggestions for the reviewer

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- GitHub Personal Access Token

### 1. Clone and Setup

```bash
git clone <repo-url>
cd project1
cp .env.example .env
# Edit .env with your GitHub token and webhook secret
```

### 2. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

### 3. Run with Docker Compose

```bash
cd docker
docker compose up -d
```

This starts:
- **API** at `http://localhost:8000`
- **PostgreSQL** at `localhost:5432`
- **Redis** at `localhost:6379`
- **MLflow** at `http://localhost:5000`

### 4. Run Locally (Development)

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. API Documentation

Open `http://localhost:8000/docs` for interactive Swagger UI.

---

## Project Structure

```
project1/
├── app/                          # FastAPI application
│   ├── main.py                   # App factory, middleware
│   ├── config.py                 # Pydantic Settings
│   ├── api/routes/               # API endpoints
│   │   ├── webhook.py            # GitHub webhook handler
│   │   ├── analysis.py           # Analysis CRUD
│   │   ├── health.py             # Health checks
│   │   └── dashboard.py          # Metrics dashboard
│   ├── core/                     # Business logic
│   │   ├── github_client.py      # GitHub API client
│   │   ├── diff_parser.py        # Unified diff parser
│   │   ├── feature_engine.py     # Feature engineering (30+ features)
│   │   └── security.py           # Auth & signature verification
│   ├── ml/                       # ML inference
│   │   ├── model.py              # XGBoost + CodeBERT models
│   │   └── risk_report.py        # GitHub comment formatter
│   ├── models/                   # SQLAlchemy ORM
│   ├── tasks/                    # Celery async tasks
│   └── db/                       # Database session management
├── ml_pipeline/                  # Training pipeline
│   ├── data_collection/          # Repository mining
│   └── training/                 # Model training scripts
├── tests/                        # Test suite
├── docker/                       # Docker configuration
├── pyproject.toml                # Dependencies & config
└── .env.example                  # Environment template
```

---

## Architecture

```
GitHub PR Event
      │
      ▼
  Webhook Receiver (FastAPI)
      │
      ▼
  Task Queue (Celery + Redis)
      │
      ▼
  ┌─────────────────────┐
  │  Analysis Pipeline   │
  │  1. Fetch PR data    │
  │  2. Parse diff       │
  │  3. Extract features │
  │  4. ML inference     │
  │  5. Generate report  │
  └─────────────────────┘
      │
      ▼
  Post Comment on PR
```

---

## ML Models

### Baseline (MVP)
- **XGBoost** with 30+ handcrafted features
- Features: change metrics, code complexity, text signals, entropy

### Transformer (v2)
- **CodeBERT** (microsoft/codebert-base) fine-tuned for classification
- Input: `[CLS] commit_message [SEP] diff_tokens [SEP]`

### Training

```bash
# Train baseline models
python -m ml_pipeline.training.train_baseline \
    --data-path data/processed/features.parquet \
    --output-dir models/baseline

# Fine-tune CodeBERT
python -m ml_pipeline.training.train_codebert \
    --data-path data/processed/tokenized_diffs.parquet \
    --output-dir models/codebert-v1
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

## GitHub Webhook Setup

1. Go to your repo → **Settings** → **Webhooks** → **Add webhook**
2. **Payload URL**: `https://your-domain.com/webhook/`
3. **Content type**: `application/json`
4. **Secret**: Same value as `GITHUB_WEBHOOK_SECRET` in your `.env`
5. **Events**: Select "Pull requests"

For local testing, use [ngrok](https://ngrok.com/):
```bash
ngrok http 8000
# Use the ngrok URL as your webhook Payload URL
```

---

## Technologies

| Category | Stack |
|----------|-------|
| **Backend** | FastAPI, Uvicorn, Celery, Redis |
| **ML** | PyTorch, HuggingFace Transformers, XGBoost, scikit-learn |
| **Database** | PostgreSQL, SQLAlchemy, Alembic |
| **MLOps** | MLflow, Docker, GitHub Actions |
| **Monitoring** | Prometheus, Grafana, structlog |

---

## License

MIT
