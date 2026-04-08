"""
Ensemble Model for Money Laundering Detection

Combines predictions from:
- Random Forest (baseline)
- GraphSAGE
- Temporal Graph Network (TGN)

Uses learned weights and calibration for optimal combination.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import torch
from torch import nn
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class EnsembleResult:
    metrics: dict[str, float]
    model_weights: dict[str, float]
    individual_metrics: dict[str, dict[str, float]]


class LearnedWeightEnsemble(nn.Module):
    """Ensemble that learns optimal combination weights."""
    
    def __init__(self, num_models: int = 3):
        super().__init__()
        # Learnable weights initialized uniformly
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        # Optional: small MLP for non-linear combination
        self.combine_mlp = nn.Sequential(
            nn.Linear(num_models, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        # Blend parameter between weighted average and MLP
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """Combine model predictions.
        
        Args:
            predictions: (batch, num_models) - probability predictions from each model
        
        Returns:
            combined: (batch,) - ensemble probability
        """
        # Weighted average
        weights = torch.softmax(self.weights, dim=0)
        weighted_avg = (predictions * weights).sum(dim=1)
        
        # MLP combination
        mlp_pred = self.combine_mlp(predictions).squeeze(-1)
        
        # Blend
        blend = torch.sigmoid(self.blend)
        return blend * weighted_avg + (1 - blend) * mlp_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


class EnsemblePredictor:
    """Production-ready ensemble predictor with calibration."""
    
    def __init__(
        self,
        baseline_model,
        graphsage_model,
        tgn_model,
        ensemble_weights: dict[str, float] | None = None,
        calibrator: IsotonicRegression | None = None,
    ):
        self.baseline_model = baseline_model
        self.graphsage_model = graphsage_model
        self.tgn_model = tgn_model
        
        # Default weights if not provided
        self.weights = ensemble_weights or {
            "baseline": 0.2,
            "graphsage": 0.3,
            "tgn": 0.5,
        }
        
        self.calibrator = calibrator
    
    def predict_proba(
        self,
        baseline_features: np.ndarray,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> dict[str, float]:
        """Get probability predictions from all models and combine.
        
        Returns dict with individual and combined probabilities.
        """
        # Baseline prediction
        baseline_prob = float(self.baseline_model.predict_proba(baseline_features)[0, 1])
        
        # GraphSAGE prediction
        self.graphsage_model.eval()
        with torch.no_grad():
            from graph.temporal_graph import GraphBatch
            batch = GraphBatch(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                labels=torch.zeros(edge_index.size(1)),
                edge_ids=torch.arange(edge_index.size(1)),
            )
            logits, _ = self.graphsage_model(batch)
            graphsage_prob = float(torch.sigmoid(logits[0]).item())
        
        # TGN prediction
        self.tgn_model.eval()
        with torch.no_grad():
            logits, _, _ = self.tgn_model(
                node_features,
                edge_index,
                edge_attr,
            )
            tgn_prob = float(torch.sigmoid(logits[0]).item())
        
        # Weighted combination
        combined = (
            self.weights["baseline"] * baseline_prob +
            self.weights["graphsage"] * graphsage_prob +
            self.weights["tgn"] * tgn_prob
        )
        
        # Calibrate if calibrator available
        if self.calibrator is not None:
            combined = float(self.calibrator.predict([[combined]])[0])
        
        return {
            "baseline_probability": round(baseline_prob, 4),
            "graphsage_probability": round(graphsage_prob, 4),
            "tgn_probability": round(tgn_prob, 4),
            "ensemble_probability": round(combined, 4),
        }


def train_ensemble(
    baseline_probs: np.ndarray,
    graphsage_probs: np.ndarray,
    tgn_probs: np.ndarray,
    y_true: np.ndarray,
    artifact_dir: str | Path,
    epochs: int = 100,
) -> tuple[LearnedWeightEnsemble, dict[str, float], IsotonicRegression]:
    """Train the ensemble weights and calibrator.
    
    Args:
        baseline_probs: Baseline model probabilities
        graphsage_probs: GraphSAGE probabilities
        tgn_probs: TGN probabilities
        y_true: True labels
        artifact_dir: Directory to save artifacts
        epochs: Training epochs for weight learning
    
    Returns:
        ensemble: Trained ensemble model
        optimal_weights: Optimized model weights
        calibrator: Fitted calibration model
    """
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    
    # Stack predictions
    all_probs = np.stack([baseline_probs, graphsage_probs, tgn_probs], axis=1)
    
    # Convert to tensors
    probs_tensor = torch.tensor(all_probs, dtype=torch.float32)
    labels_tensor = torch.tensor(y_true, dtype=torch.float32)
    
    # Initialize ensemble
    ensemble = LearnedWeightEnsemble(num_models=3)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        ensemble.train()
        optimizer.zero_grad()
        
        combined_probs = ensemble(probs_tensor)
        loss = criterion(combined_probs, labels_tensor)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        ensemble.eval()
        with torch.no_grad():
            probs = ensemble(probs_tensor).numpy()
            preds = (probs >= 0.5).astype(int)
            current_f1 = f1_score(y_true, preds, zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = {k: v.clone() for k, v in ensemble.state_dict().items()}
    
    # Load best
    if best_state is not None:
        ensemble.load_state_dict(best_state)
    
    # Get optimal weights
    ensemble.eval()
    with torch.no_grad():
        weights = torch.softmax(ensemble.weights, dim=0).numpy()
    
    optimal_weights = {
        "baseline": float(weights[0]),
        "graphsage": float(weights[1]),
        "tgn": float(weights[2]),
    }
    
    # Get combined predictions for calibration
    with torch.no_grad():
        combined_probs = ensemble(probs_tensor).numpy()
    
    # Fit calibrator
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(combined_probs, y_true)
    
    # Save artifacts
    torch.save(ensemble.state_dict(), artifact_root / "ensemble_weights.pt")
    with open(artifact_root / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(artifact_root / "optimal_weights.json", "w") as f:
        json.dump(optimal_weights, f, indent=2)
    
    return ensemble, optimal_weights, calibrator


def evaluate_ensemble(
    baseline_probs: np.ndarray,
    graphsage_probs: np.ndarray,
    tgn_probs: np.ndarray,
    y_true: np.ndarray,
    weights: dict[str, float],
    calibrator: IsotonicRegression | None = None,
) -> EnsembleResult:
    """Evaluate the ensemble on test data.
    
    Args:
        baseline_probs: Baseline probabilities
        graphsage_probs: GraphSAGE probabilities
        tgn_probs: TGN probabilities
        y_true: True labels
        weights: Model weights
        calibrator: Optional calibrator
    
    Returns:
        EnsembleResult with metrics
    """
    # Individual model metrics
    individual_metrics = {}
    
    for name, probs in [("baseline", baseline_probs), ("graphsage", graphsage_probs), ("tgn", tgn_probs)]:
        preds = (probs >= 0.5).astype(int)
        individual_metrics[name] = compute_metrics(y_true, preds, probs)
    
    # Weighted combination
    combined = (
        weights["baseline"] * baseline_probs +
        weights["graphsage"] * graphsage_probs +
        weights["tgn"] * tgn_probs
    )
    
    # Calibrate
    if calibrator is not None:
        combined = calibrator.predict(combined)
    
    # Ensemble metrics
    preds = (combined >= 0.5).astype(int)
    ensemble_metrics = compute_metrics(y_true, preds, combined)
    
    return EnsembleResult(
        metrics=ensemble_metrics,
        model_weights=weights,
        individual_metrics=individual_metrics,
    )
