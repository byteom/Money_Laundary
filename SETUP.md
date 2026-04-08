# Setup Guide - Advanced AML Detection System

Complete installation and setup guide for the Money Laundering Detection system using Temporal Graph Networks.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Models](#training-models)
5. [Running the API Server](#running-the-api-server)
6. [Testing](#testing)
7. [AWS Deployment](#aws-deployment)
8. [Troubleshooting](#troubleshooting)

---

## 🌐 Live Demo

**The system is deployed on AWS and accessible at:**

| Resource | URL |
|----------|-----|
| **Web Interface** | http://13.126.159.27/ |
| **Health Check** | http://13.126.159.27/health |
| **Prediction API** | http://13.126.159.27/predict |

For detailed AWS deployment instructions, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

---

## System Requirements

### Hardware
- **CPU:** Minimum 4 cores (8+ recommended)
- **RAM:** 8GB minimum (16GB recommended for large datasets)
- **Storage:** 2GB free space

### Software
- **Python:** 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues with older PyTorch versions)
- **Operating System:** Windows, macOS, or Linux
- **Git:** For cloning the repository (optional)

---

## Installation

### Step 1: Clone or Download the Project

**Option A: Clone with Git**
```bash
git clone <repository-url>
cd Money_Laundary
```

**Option B: Download ZIP**
1. Download the project ZIP file
2. Extract to a folder
3. Navigate to the `Money_Laundary` folder

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages:**
- `torch` (PyTorch for deep learning)
- `torch-geometric` (Graph neural network library)
- `scikit-learn` (Machine learning utilities)
- `pandas` (Data manipulation)
- `numpy` (Numerical computing)
- `flask` (Web API framework)
- `flask-cors` (CORS support)
- `joblib` (Model serialization)
- `pytest` (Testing framework)

**Installation time:** 5-10 minutes depending on internet speed.

### Step 4: Verify Installation

```bash
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

If you see "Installation successful!", you're ready to proceed!

---

## Dataset Preparation

### Step 1: Download Dataset

**Option A: Use Provided Synthetic Dataset**
- Dataset is already in `data/synthetic_transaction_data.csv`
- Contains 100,000 transactions with fraud labels
- Ready to use immediately

**Option B: Use Custom Dataset**
If using your own dataset, ensure it has these columns:
- `sender_id` (string)
- `receiver_id` (string)
- `transaction_amount` (float)
- `timestamp` (datetime string)
- `transaction_type` (string)
- `is_fraud` (0 or 1)

Place your dataset as `data/your_data.csv`

### Step 2: Verify Dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('data/synthetic_aml_transactions.csv'); print(df.head())"
```

### Step 3: Run Preprocessing

```bash
python preprocessing/preprocess.py
```

**What this does:**
- Loads raw transaction data
- Computes 31 advanced node features (PageRank, velocity, structuring patterns)
- Engineers 75 transaction features
- Builds temporal graph structure
- Creates train/test split (80/20)
- Saves processed data to `data/processed/`

**Expected output files:**
- `data/processed/graph.pkl` - Graph structure
- `data/processed/node_features.npy` - Node embeddings
- `data/processed/train_data.pkl` - Training set
- `data/processed/test_data.pkl` - Test set
- `data/processed/scaler.pkl` - Feature scaler

**Processing time:** 2-5 minutes for 100K transactions

---

## Training Models

### Full Training Pipeline

Train all models (Logistic Regression, Random Forest, GraphSAGE, TGN, and Ensemble):

```bash
cd training
python train.py
```


**Training process:**
1. **Phase 1:** Train baseline models (LR + RF) - ~30 seconds
2. **Phase 2:** Train GraphSAGE - ~2 minutes
3. **Phase 3:** Train TGN with Time2Vec - ~3 minutes
4. **Phase 4:** Create ensemble with learned weights - ~1 minute
5. **Phase 5:** Generate performance report

**Total training time:** ~7-10 minutes

**Expected console output:**
```
========================================
TRAINING PIPELINE
========================================

[1/5] Training Baseline Models...
  - Logistic Regression: F1=0.74
  - Random Forest: F1=0.81

[2/5] Training GraphSAGE...
Epoch 50: Loss=0.245, Val F1=0.87

[3/5] Training TGN...
Epoch 79: Loss=0.198, Val F1=0.82

[4/5] Building Ensemble...
Ensemble F1: 0.92

[5/5] Saving Models and Report...
Report saved to: ../models/model_comparison_report.md
```

### Trained Model Files

After training, you'll find these files in `artifacts/models/`:
- `logistic_regression.joblib` - Baseline logistic regression
- `random_forest.joblib` - Baseline random forest
- `temporal_graphsage.pt` - GraphSAGE neural network
- `tgn_model.pt` - Temporal Graph Network
- `ensemble_weights.pt` - Learned ensemble weights
- `calibrator.pkl` - Isotonic regression calibrator
- `optimal_weights.json` - Ensemble weight values
- `gnn_node_embeddings.pkl` - Node embeddings
- `baseline_metrics.json` - Baseline model metrics
- `gnn_metrics.json` - GNN metrics
- `tgn_metrics.json` - TGN metrics

And the report in `artifacts/reports/`:
- `project_report.md` - Full performance report

### Training Individual Models (Optional)

**Train only GraphSAGE:**
```bash
cd models
python graphsage.py
```

**Train only TGN:**
```bash
cd models
python tgn.py
```

---

## Running the API Server

### Step 1: Start Flask Server

```bash
cd deployment
python app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
 * Loading models...
 * All models loaded successfully
 * Ready to accept requests
```

### Step 2: Access Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the advanced web interface with:
- Transaction input form
- Pre-configured test scenarios
- Real-time fraud risk analysis
- Model comparison dashboard

### Step 3: Test API Endpoints

**Health Check:**
```bash
curl http://localhost:5000/
```

**Get Model Information:**
```bash
curl http://localhost:5000/model-info
```

**Make Prediction (PowerShell):**
```powershell
$body = @{
    sender_id = "ACC_0001"
    receiver_id = "ACC_0002"
    transaction_amount = 9500.00
    timestamp = "2024-01-15 14:30:00"
    transaction_type = "transfer"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/predict -Method POST -Body $body -ContentType "application/json"
```

**Make Prediction (Linux/Mac):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "ACC_0001",
    "receiver_id": "ACC_0002",
    "transaction_amount": 9500.00,
    "timestamp": "2024-01-15 14:30:00",
    "transaction_type": "transfer"
  }'
```

**Expected JSON response:**
```json
{
  "baseline_probability": 0.65,
  "graphsage_probability": 0.78,
  "tgn_probability": 0.72,
  "ensemble_probability": 0.75,
  "risk_classification": "high",
  "model_weights": {
    "baseline": 0.15,
    "graphsage": 0.45,
    "tgn": 0.40
  }
}
```

### Stopping the Server

Press `Ctrl+C` in the terminal running the Flask app.

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

**Test suite includes:**
- **Unit tests:** Individual model components (Time2Vec, Focal Loss, Memory Module)
- **Integration tests:** Full pipeline validation
- **Model tests:** Forward pass, gradient flow

**Expected output:**
```
tests/test_tgn.py::TestTime2Vec::test_time2vec_output_shape PASSED
tests/test_tgn.py::TestFocalLoss::test_focal_loss_reduces_easy_examples PASSED
...
======================== 24 passed in 45.2s ========================
```

### Run Specific Test Modules

**Test TGN components only:**
```bash
pytest tests/test_tgn.py -v
```

**Test integration only:**
```bash
pytest tests/test_integration.py -v
```

### Test Coverage (Optional)

```bash
pip install pytest-cov
pytest --cov=. --cov-report=html tests/
```

Open `htmlcov/index.html` to view coverage report.

---

## Troubleshooting

### Common Issues

#### 1. **ImportError: No module named 'torch_geometric'**

**Solution:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

*(Replace `torch-2.0.0+cpu` with your PyTorch version and CUDA version)*

#### 2. **FileNotFoundError: data/processed/graph.pkl not found**

**Solution:** Run preprocessing first
```bash
cd preprocessing
python preprocess.py
```

#### 3. **Flask server won't start: "Address already in use"**

**Solution:** Kill existing process on port 5000

**Windows:**
```powershell
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -ti:5000 | xargs kill -9
```

#### 4. **Low GPU memory / CUDA out of memory**

**Solution:** Edit `training/train.py` and reduce batch size:
```python
batch_size = 64  # Change to 32 or 16
```

Or force CPU mode:
```python
device = torch.device('cpu')
```

#### 5. **pandas errors about frequency strings**

**Solution:** Update pandas to 2.0+:
```bash
pip install --upgrade pandas
```

#### 6. **Tests fail with "no module named tests"**

**Solution:** Run pytest from project root:
```bash
cd C:\Users\anwee\Downloads\Money_Laundary\Money_Laundary
pytest tests/ -v
```

#### 7. **Slow training on CPU**

**Expected behavior:** Training on CPU takes ~10 minutes. For faster training:
- Install CUDA-enabled PyTorch
- Use a GPU with at least 4GB VRAM
- See: https://pytorch.org/get-started/locally/

---

## Project Structure

```
Money_Laundary/
├── data/
│   ├── synthetic_transaction_data.csv  # Raw data
│   └── processed/                      # Processed files
├── preprocessing/
│   └── preprocess.py                   # Feature engineering
├── models/
│   ├── baseline.py                     # Logistic Regression, Random Forest
│   ├── graphsage.py                    # GraphSAGE model
│   ├── tgn.py                          # Temporal GNN with Time2Vec
│   ├── ensemble.py                     # Ensemble predictor
│   └── *.pth, *.pkl                    # Trained model files
├── training/
│   └── train.py                        # Full training pipeline
├── deployment/
│   ├── app.py                          # Flask API
│   └── static/
│       └── index.html                  # Web interface
├── tests/
│   ├── test_tgn.py                     # TGN unit tests
│   └── test_integration.py             # Integration tests
├── requirements.txt                    # Dependencies
├── README.md                           # Project overview
├── SETUP.md                            # This file
└── PROJECT_OVERVIEW.md                 # Detailed project explanation
```

---

## Next Steps

After successful setup:

1. **Explore the Web UI:** Test different transaction scenarios
2. **Review Model Report:** Open `models/model_comparison_report.md`
3. **Read Project Overview:** See `PROJECT_OVERVIEW.md` for architecture details
4. **Customize Features:** Edit `preprocessing/preprocess.py` to add domain-specific features
5. **Tune Hyperparameters:** Modify `training/train.py` for better performance

---

## Getting Help

**Check logs:**
- Training logs: Console output from `train.py`
- API logs: Flask console
- Test logs: Pytest output with `-v` flag

**Documentation:**
- Main README: `README.md`
- Project Overview: `PROJECT_OVERVIEW.md`
- Code comments: All modules are documented

**Common commands reference:**
```bash
# Activate environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Preprocess data
cd preprocessing && python preprocess.py

# Train models
cd training && python train.py

# Run tests
pytest tests/ -v

# Start API
cd deployment && python app.py
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Ensemble F1 Score | ~0.92 |
| TGN F1 Score | ~0.82 |
| GraphSAGE F1 Score | ~0.89 |
| ROC-AUC | ~0.99 |
| Training Time | 7-10 minutes (CPU) |
| Inference Time | <50ms per transaction |

---

## System Tested On

✅ **Windows 10/11** - Python 3.9  
✅ **Ubuntu 20.04/22.04** - Python 3.8, 3.9  
✅ **macOS 12+** - Python 3.9, 3.10  

---

**Setup Complete!** 🎉

You now have a fully functional advanced money laundering detection system. Explore the web interface and experiment with different transaction patterns.
