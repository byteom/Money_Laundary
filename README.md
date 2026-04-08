# Advanced Money Laundering Detection Using Temporal Graph Networks

This project implements a **state-of-the-art** anti-money-laundering (AML) detection system using temporal graph neural networks with advanced feature engineering and ensemble learning.

## 🌟 Key Features

### Advanced Model Architecture
- **Temporal Graph Network (TGN)**: Full implementation with memory modules, temporal attention, and Time2Vec encoding
- **GraphSAGE**: Message-passing neural network for graph-based learning
- **Ensemble Model**: Learned weight combination with probability calibration

### AML-Specific Feature Engineering
- **Structuring Detection**: Near-threshold amounts, round number frequency
- **Velocity Analysis**: Transaction rate, burst detection, dormancy patterns
- **Graph Metrics**: PageRank, cycle participation, flow imbalance
- **Temporal Windows**: Rolling 1h/6h/24h aggregations

### Production-Ready Deployment
- Flask API with multiple model predictions
- Risk classification (critical/high/medium/low/minimal)
- Model explainability through attention weights

## Project Structure

```text
data/               Dataset generation
preprocessing/      Cleaning, encoding, scaling, advanced feature engineering
graph/              Temporal graph helpers
models/
  ├── baseline.py   Logistic Regression & Random Forest
  ├── gnn.py        Temporal GraphSAGE
  ├── tgn.py        Temporal Graph Network (TGN)
  └── ensemble.py   Weighted ensemble with calibration
training/           Training and evaluation pipeline
deployment/         Flask API for inference
tests/              Unit and integration tests
artifacts/          Processed data, trained models, and reports
```

## Model Architecture

### Temporal Graph Network (TGN)
```
Input → Time2Vec Encoding → Memory Lookup → Temporal Attention → Message Aggregation → Memory Update → Edge Classification
```

**Components:**
- **Time2Vec**: Learnable time encoding capturing periodic and linear patterns
- **Memory Module**: GRU-based state tracking for each node
- **Temporal Attention**: Multi-head attention with time decay weighting
- **Focal Loss**: Better handling of class imbalance

### Advanced Features

| Category | Features |
|----------|----------|
| Graph Structure | PageRank, cycle participation, flow imbalance |
| Velocity | Transaction rate/hour, burst score, dormancy ratio |
| Structuring | Near-threshold ratio, round number frequency |
| Temporal | 1h/6h/24h max counts, volumes, counterparties |

## Setup

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## Train All Models

```bash
# Full training (baseline + GraphSAGE + TGN + ensemble)
python -m training.train --generate

# Quick training (skip TGN for faster iteration)
python -m training.train --generate --skip-tgn
```

**Training Output:**
- Baseline models (Logistic Regression, Random Forest)
- Temporal GraphSAGE
- Temporal Graph Network (TGN)
- Ensemble with learned weights
- Model comparison report

## Run the API

```bash
python -m deployment.app
```

**Endpoints:**
- `GET /` - Web UI
- `GET /health` - Health check
- `POST /predict` - Fraud prediction
- `GET /model-info` - Model information

## Example Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "ACC_0001",
    "receiver_id": "ACC_0002",
    "transaction_amount": 9500,
    "timestamp": "2025-03-12T10:15:00",
    "transaction_type": "transfer"
  }'
```

**Response:**
```json
{
  "baseline_probability": 0.23,
  "graphsage_probability": 0.31,
  "tgn_probability": 0.42,
  "ensemble_probability": 0.35,
  "fraud_probability": 0.35,
  "risk_classification": "low",
  "model_weights": {
    "baseline": 0.2,
    "graphsage": 0.3,
    "tgn": 0.5
  }
}
```

## Dataset

The training pipeline generates synthetic AML data with:
- Normal transactions (transfer, payment, deposit, withdrawal, cash_out)
- Suspicious patterns (rapid chains, layered movement, structuring)

**Features:**
- `sender_id`, `receiver_id` - Account identifiers
- `transaction_amount` - Transaction value
- `timestamp` - Transaction time
- `transaction_type` - Transaction category
- `label` - Fraud indicator (0/1)

## Training Outputs

After training, the project generates:

```
artifacts/
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── temporal_graphsage.pt
│   ├── tgn_model.pt
│   ├── ensemble_weights.pt
│   ├── calibrator.pkl
│   └── optimal_weights.json
├── processed/
│   ├── train_transactions.csv
│   ├── test_transactions.csv
│   ├── node_features.csv
│   └── feature_metadata.json
└── reports/
    ├── model_comparison.csv
    ├── project_report.md
    └── summary.json
```

## Expected Performance

| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~0.74 | ~0.98 |
| Random Forest | ~0.81 | ~0.99 |
| GraphSAGE | ~0.87 | ~0.99 |
| **TGN** | ~0.90 | ~0.99 |
| **Ensemble** | ~0.92 | ~0.99 |

## Technical Highlights

1. **No External GNN Libraries**: Pure PyTorch implementation for portability
2. **Focal Loss**: Addresses class imbalance in fraud detection
3. **Isotonic Calibration**: Ensures well-calibrated probability outputs
4. **Modular Design**: Easy to extend with new features or models
5. **Comprehensive Testing**: Unit and integration test coverage

## Assumptions & Limitations

- Uses synthetic data (real AML data is typically confidential)
- Graph context limited by available transaction history
- Near-threshold detection assumes standard reporting thresholds

## License

This project is for educational and demonstration purposes.

