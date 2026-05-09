# Advanced AML Detection Project Report

## Executive Summary
This project implements a state-of-the-art money laundering detection system using
temporal graph neural networks with advanced feature engineering.

## Model Performance Comparison

### Baseline Models
- **Best baseline**: random_forest
  - F1 Score: 0.8655
  - ROC-AUC: 0.9941
  - Precision: 0.9084
  - Recall: 0.8264

### Graph Neural Network (GraphSAGE)
- F1 Score: 0.8970
- ROC-AUC: 0.9953
- Precision: 0.8599
- Recall: 0.9375

### Temporal Graph Network (TGN) - Advanced
- F1 Score: 0.7946
- ROC-AUC: 0.9769
- Precision: 0.7712
- Recall: 0.8194

### Ensemble Model (Best)
- F1 Score: 0.9342
- ROC-AUC: 0.9980
- Precision: 0.8875
- Recall: 0.9861

## Advanced Features Implemented

### Graph Structure Features
- PageRank scores for node importance
- Cycle participation detection
- Flow imbalance analysis

### Velocity & Burst Detection
- Transaction velocity per hour
- Burst score (max hourly vs average)
- Dormancy ratio detection
- Activity concentration (Gini-like measure)

### Structuring Detection (AML-specific)
- Near-threshold transaction ratio
- Round number frequency
- Amount variance analysis
- Combined structuring score

### Temporal Window Features
- Rolling 1h, 6h, 24h aggregations
- Max transaction count per window
- Max volume per window
- Unique counterparties per window

## Architecture Highlights

### Temporal Graph Network (TGN)
- **Time2Vec**: Learnable time encoding capturing periodic and linear patterns
- **Memory Module**: GRU-based state tracking for each node
- **Temporal Attention**: Multi-head attention with time decay weighting
- **Focal Loss**: Better handling of class imbalance

### Ensemble Strategy
- Learned weight combination of Random Forest, GraphSAGE, and TGN
- Isotonic regression calibration for probability outputs

## Interpretation Notes

- Baseline models rely on handcrafted sender/receiver and time features
- GraphSAGE adds neighborhood context through message passing
- TGN incorporates temporal memory and attention mechanisms
- The ensemble combines complementary strengths of all models

### Expected Error Patterns
- **False Positives**: Bursty legitimate high-volume payment chains
- **False Negatives**: Isolated suspicious transfers with weak graph context