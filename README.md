# Advanced AML Detection with Temporal Graph Networks

An end-to-end Anti-Money Laundering (AML) detection system that combines:
- Baseline ML models (Logistic Regression, Random Forest)
- Graph neural networks (Temporal GraphSAGE)
- A Temporal Graph Network (TGN) with memory and time encoding
- A learned weighted ensemble for final fraud risk scoring

The project includes data generation, preprocessing, model training, evaluation, tests, and a Flask API + web UI for inference.

## Table of Contents

1. [What This Project Solves](#what-this-project-solves)
2. [Repository Layout](#repository-layout)
3. [Tech Stack](#tech-stack)
4. [Quick Start](#quick-start)
5. [Detailed Setup](#detailed-setup)
6. [Data Pipeline](#data-pipeline)
7. [Training Pipeline](#training-pipeline)
8. [Run the API](#run-the-api)
9. [API Reference](#api-reference)
10. [Model Performance](#model-performance)
11. [Testing](#testing)
12. [Artifacts Generated](#artifacts-generated)
13. [Deployment Notes](#deployment-notes)
14. [Troubleshooting](#troubleshooting)
15. [Limitations and Assumptions](#limitations-and-assumptions)

## What This Project Solves

Traditional AML systems often miss complex behavior because they do not model:
- Account-to-account relationships as a graph
- Temporal dynamics (rapid chains, bursty activity, dormancy)
- Structuring patterns (transactions near reporting thresholds)

This project addresses those gaps by combining engineered AML features with temporal graph learning.

## Repository Layout

```text
Money_Laundary/
  data/
    generate_synthetic_data.py      # Synthetic transaction generator
    synthetic_aml_transactions.csv  # Default dataset
  preprocessing/
    preprocess.py                    # Feature engineering + graph prep
  graph/
    temporal_graph.py                # Graph batch utilities
  models/
    baseline.py                      # Logistic Regression + Random Forest
    gnn.py                           # Temporal GraphSAGE model and training
    tgn.py                           # Temporal Graph Network model and training
    ensemble.py                      # Learned weighted ensemble + calibration
  training/
    train.py                         # End-to-end training pipeline
  deployment/
    app.py                           # Flask API
    static/index.html                # Web UI
  tests/
    test_tgn.py
    test_integration.py
  artifacts/
    models/                          # Saved models and weights
    processed/                       # Processed data and preprocessors
    reports/                         # Evaluation reports
```

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- PyTorch
- Flask + Flask-CORS
- Pytest

See [requirements.txt](requirements.txt) for exact dependencies.

## Quick Start

From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train all models (uses `data/synthetic_aml_transactions.csv` by default):

```bash
python -m training.train
```

Start API server:

```bash
python -m deployment.app
```

Open:
- `http://127.0.0.1:5000/` for the web UI
- `http://127.0.0.1:5000/health` for health check

## Detailed Setup

### Prerequisites

- Python 3.8+
- 8 GB RAM minimum (16 GB recommended for larger experiments)
- Optional: virtual environment tool (`venv`)

### Install

```bash
python -m pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch, sklearn, flask, pandas; print('ok')"
```

## Data Pipeline

### 1. Generate synthetic data (optional)

If you want to regenerate the dataset:

```bash
python data/generate_synthetic_data.py
```

This creates/overwrites:
- `data/synthetic_aml_transactions.csv`

### 2. Preprocessing and feature engineering

You can run preprocessing directly:

```bash
python -c "from preprocessing.preprocess import prepare_dataset; from pathlib import Path; prepare_dataset(Path('data/synthetic_aml_transactions.csv'), Path('artifacts/processed'))"
```

In normal use, preprocessing is run automatically inside `training/train.py`.

### Engineered features include

- Graph structure: PageRank, cycle participation, flow imbalance
- Velocity: transaction rate, burst score, dormancy ratio
- Structuring signals: near-threshold ratio, round-number behavior
- Temporal windows: rolling 1h/6h/24h activity stats

## Training Pipeline

Run full training:

```bash
python -m training.train
```

Useful flags:

```bash
python -m training.train --skip-tgn
python -m training.train --dataset data/synthetic_aml_transactions.csv --artifacts artifacts
```

What training does:

1. Loads and preprocesses data
2. Trains baseline models
3. Trains Temporal GraphSAGE
4. Trains TGN (unless `--skip-tgn`)
5. Learns ensemble weights and optional probability calibration
6. Saves models and reports under `artifacts/`

## Run the API

Start locally:

```bash
python -m deployment.app
```

The app loads:
- preprocessors from `artifacts/processed/preprocessors.pkl`
- model assets from `artifacts/models/`

If required artifact files are missing, train first with `python -m training.train`.

## API Reference

### `GET /health`
Returns service status.

### `GET /model-info`
Returns available models, ensemble weights, risk thresholds, and feature summary.

### `POST /predict`
Request body:

```json
{
  "sender_id": "ACC_0001",
  "receiver_id": "ACC_0002",
  "transaction_amount": 9500,
  "timestamp": "2025-03-12T10:15:00",
  "transaction_type": "transfer"
}
```

Required fields:
- `sender_id`
- `receiver_id`
- `transaction_amount`
- `timestamp` (ISO-8601 compatible)
- `transaction_type` (must match known classes)

Response shape:

```json
{
  "baseline_probability": 0.23,
  "graphsage_probability": 0.31,
  "tgn_probability": 0.42,
  "ensemble_probability": 0.35,
  "fraud_probability": 0.35,
  "risk_classification": "low",
  "model_weights": {
    "baseline": 0.4239,
    "graphsage": 0.4834,
    "tgn": 0.0926
  }
}
```

Risk classification thresholds (from `deployment/app.py`):
- `critical`: $p \ge 0.85$
- `high`: $0.70 \le p < 0.85$
- `medium`: $0.45 \le p < 0.70$
- `low`: $0.25 \le p < 0.45$
- `minimal`: $p < 0.25$

## Model Performance

Latest summary from `artifacts/reports/summary.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9500 | 0.7165 | 0.9653 | 0.8225 | 0.9896 |
| Random Forest | 0.9692 | 0.9084 | 0.8264 | 0.8655 | 0.9941 |
| Temporal GraphSAGE | 0.9742 | 0.9124 | 0.8681 | 0.8897 | 0.9950 |
| TGN | 0.9542 | 0.7908 | 0.8403 | 0.8148 | 0.9754 |
| Ensemble | 0.9783 | 0.8933 | 0.9306 | 0.9116 | 0.9945 |

Learned ensemble weights:
- baseline: 0.42396
- graphsage: 0.48343
- tgn: 0.09261

## Testing

Run all tests:

```bash
python -m pytest tests -v
```

Run only integration tests:

```bash
python -m pytest tests/test_integration.py -v
```

Run only TGN tests:

```bash
python -m pytest tests/test_tgn.py -v
```

## Artifacts Generated

After training, check these paths:

- `artifacts/models/`
  - `logistic_regression.joblib`
  - `random_forest.joblib`
  - `temporal_graphsage.pt`
  - `tgn_model.pt`
  - `optimal_weights.json`
  - `calibrator.pkl`
  - `gnn_node_embeddings.pkl`
- `artifacts/processed/`
  - `train_transactions.csv`
  - `test_transactions.csv`
  - `node_features.csv`
  - `feature_metadata.json`
  - `preprocessors.pkl`
- `artifacts/reports/`
  - `model_comparison.csv`
  - `project_report.md`
  - `summary.json`

## Deployment Notes

The repository also includes:
- [SETUP.md](SETUP.md) for step-by-step setup details
- [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for cloud deployment notes
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture and rationale

## Troubleshooting

- `FileNotFoundError` for model/preprocessor files:
  - Run `python -m training.train` first.
- `Unsupported transaction_type` from `/predict`:
  - Use one of the transaction types learned during preprocessing.
- Import/module path errors:
  - Run commands from the repository root.
- Slow training on CPU:
  - Use `--skip-tgn` for a faster iteration loop.

## Limitations and Assumptions

- The project uses synthetic data, not confidential real bank data.
- Performance can vary by random seed and data generation pattern.
- Graph context depends on available transaction history.

## License

This project is intended for educational and demonstration use.
