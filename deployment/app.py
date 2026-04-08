from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from models.gnn import TemporalGraphSAGE
from models.tgn import TemporalGraphNetwork


ARTIFACT_ROOT = Path("artifacts")
MODELS_DIR = ARTIFACT_ROOT / "models"
PROCESSED_DIR = ARTIFACT_ROOT / "processed"

app = Flask(__name__, static_folder="static")
CORS(app)
ASSETS = None


def _load_assets() -> dict:
    with open(PROCESSED_DIR / "preprocessors.pkl", "rb") as handle:
        preprocessors = pickle.load(handle)
    with open(MODELS_DIR / "gnn_node_embeddings.pkl", "rb") as handle:
        embedding_assets = pickle.load(handle)
    metadata = json.loads((PROCESSED_DIR / "feature_metadata.json").read_text())

    # Load baseline model
    baseline_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    
    # Load GraphSAGE model
    gnn_model = TemporalGraphSAGE(
        node_dim=embedding_assets["node_feature_dim"],
        edge_dim=embedding_assets["edge_feature_dim"],
        hidden_dim=64,
    )
    gnn_model.load_state_dict(torch.load(MODELS_DIR / "temporal_graphsage.pt", map_location="cpu"))
    gnn_model.eval()
    
    # Load TGN model if available
    tgn_model = None
    if (MODELS_DIR / "tgn_model.pt").exists():
        tgn_model = TemporalGraphNetwork(
            node_dim=embedding_assets["node_feature_dim"],
            edge_dim=embedding_assets["edge_feature_dim"],
            memory_dim=64,
            time_dim=16,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )
        tgn_model.load_state_dict(torch.load(MODELS_DIR / "tgn_model.pt", map_location="cpu"))
        tgn_model.eval()
    
    # Load ensemble weights and calibrator if available
    ensemble_weights = {"baseline": 0.2, "graphsage": 0.3, "tgn": 0.5}
    calibrator = None
    
    if (MODELS_DIR / "optimal_weights.json").exists():
        ensemble_weights = json.loads((MODELS_DIR / "optimal_weights.json").read_text())
    
    if (MODELS_DIR / "calibrator.pkl").exists():
        with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)

    return {
        "preprocessors": preprocessors,
        "embeddings": embedding_assets,
        "metadata": metadata,
        "baseline_model": baseline_model,
        "gnn_model": gnn_model,
        "tgn_model": tgn_model,
        "ensemble_weights": ensemble_weights,
        "calibrator": calibrator,
    }


def _ensure_assets() -> dict:
    global ASSETS
    if ASSETS is None:
        ASSETS = _load_assets()
    return ASSETS


def _build_account_features(account_id: str, node_frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    match = node_frame[node_frame["account_id"] == account_id]
    if match.empty:
        return {f"{prefix}_{column}": 0.0 for column in node_frame.columns if column != "account_id"}
    row = match.iloc[0]
    return {f"{prefix}_{column}": float(row[column]) for column in node_frame.columns if column != "account_id"}


def _build_features(payload: dict, assets: dict) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    preprocessors = assets["preprocessors"]
    node_frame: pd.DataFrame = preprocessors["node_frame"]
    node_embeddings = assets["embeddings"]["node_embeddings"]
    node_matrix = preprocessors["node_matrix"]  # Raw 31-dim features for TGN
    account_to_index = preprocessors["account_to_index"]

    amount = float(payload["transaction_amount"])
    timestamp = pd.to_datetime(payload["timestamp"])
    tx_type = str(payload["transaction_type"]).strip().lower()
    tx_type_encoded = int(preprocessors["type_encoder"].transform([tx_type])[0])

    amount_scaled = preprocessors["amount_scaler"].transform(
        pd.DataFrame({"transaction_amount": [amount]})
    )[0, 0]
    gap_scaled = preprocessors["gap_scaler"].transform(
        pd.DataFrame({"sender_gap_minutes": [0.0], "receiver_gap_minutes": [0.0]})
    )[0]
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    is_weekend = int(day_of_week in [5, 6])

    sender_features = _build_account_features(payload["sender_id"], node_frame, "sender")
    receiver_features = _build_account_features(payload["receiver_id"], node_frame, "receiver")

    engineered = {
        "transaction_amount_scaled": amount_scaled,
        "transaction_type_encoded": tx_type_encoded,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend": is_weekend,
        "sender_gap_minutes_scaled": gap_scaled[0],
        "receiver_gap_minutes_scaled": gap_scaled[1],
        **sender_features,
        **receiver_features,
    }
    ordered_features = assets["metadata"]["transaction_feature_names"]
    baseline_vector = np.array([[engineered[name] for name in ordered_features]], dtype=np.float32)

    # GNN embeddings (64-dim)
    default_embedding = node_embeddings.mean(axis=0)
    sender_embedding = node_embeddings[account_to_index[payload["sender_id"]]] if payload["sender_id"] in account_to_index else default_embedding
    receiver_embedding = (
        node_embeddings[account_to_index[payload["receiver_id"]]] if payload["receiver_id"] in account_to_index else default_embedding
    )
    
    # Raw node features for TGN (31-dim)
    default_raw_features = node_matrix.mean(axis=0)
    sender_raw = node_matrix[account_to_index[payload["sender_id"]]] if payload["sender_id"] in account_to_index else default_raw_features
    receiver_raw = node_matrix[account_to_index[payload["receiver_id"]]] if payload["receiver_id"] in account_to_index else default_raw_features
    
    edge_tensor = torch.tensor(
        [
            [
                amount_scaled,
                tx_type_encoded,
                engineered["hour_sin"],
                engineered["hour_cos"],
                engineered["dow_sin"],
                engineered["dow_cos"],
                is_weekend,
                gap_scaled[0],
                gap_scaled[1],
            ]
        ],
        dtype=torch.float32,
    )

    sender_tensor = torch.tensor(np.asarray(sender_embedding).reshape(1, -1), dtype=torch.float32)
    receiver_tensor = torch.tensor(np.asarray(receiver_embedding).reshape(1, -1), dtype=torch.float32)
    raw_node_features = torch.tensor(np.vstack([sender_raw, receiver_raw]), dtype=torch.float32)
    
    return baseline_vector, sender_tensor, receiver_tensor, edge_tensor, raw_node_features


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.get("/")
def serve_frontend():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict() -> tuple:
    assets = _ensure_assets()
    payload = request.get_json(force=True)
    required_fields = {"sender_id", "receiver_id", "transaction_amount", "timestamp", "transaction_type"}
    missing = sorted(required_fields.difference(payload))
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    transaction_type = str(payload["transaction_type"]).strip().lower()
    allowed_types = {str(value).lower() for value in assets["preprocessors"]["type_encoder"].classes_}
    if transaction_type not in allowed_types:
        return (
            jsonify(
                {
                    "error": "Unsupported transaction_type.",
                    "allowed_transaction_types": sorted(allowed_types),
                }
            ),
            400,
        )
    payload["transaction_type"] = transaction_type

    try:
        datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
    except ValueError:
        return jsonify({"error": "timestamp must be ISO-8601 compatible"}), 400

    baseline_vector, sender_embedding, receiver_embedding, edge_tensor, raw_node_features = _build_features(payload, assets)
    baseline_prob = float(assets["baseline_model"].predict_proba(baseline_vector)[0, 1])

    # GraphSAGE prediction
    with torch.no_grad():
        features = torch.cat([sender_embedding, receiver_embedding, edge_tensor], dim=1)
        logits = assets["gnn_model"].edge_mlp(features).squeeze(1)
        gnn_prob = float(torch.sigmoid(logits)[0].item())

    # TGN prediction (if available)
    tgn_prob = gnn_prob  # Default to GNN if TGN not available
    if assets["tgn_model"] is not None:
        with torch.no_grad():
            # Use raw node features (31-dim) for TGN
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)
            logits, _, _ = assets["tgn_model"](raw_node_features, edge_index, edge_tensor)
            tgn_prob = float(torch.sigmoid(logits[0]).item())

    # Ensemble combination with learned weights
    weights = assets["ensemble_weights"]
    ensemble_prob = (
        weights["baseline"] * baseline_prob +
        weights["graphsage"] * gnn_prob +
        weights["tgn"] * tgn_prob
    )
    
    # Calibrate if calibrator available
    if assets["calibrator"] is not None:
        ensemble_prob = float(assets["calibrator"].predict([[ensemble_prob]])[0])
    
    ensemble_prob = round(ensemble_prob, 4)
    
    # Risk classification with more granular levels
    if ensemble_prob >= 0.85:
        risk_class = "critical"
    elif ensemble_prob >= 0.70:
        risk_class = "high"
    elif ensemble_prob >= 0.45:
        risk_class = "medium"
    elif ensemble_prob >= 0.25:
        risk_class = "low"
    else:
        risk_class = "minimal"

    return (
        jsonify(
            {
                "baseline_probability": round(baseline_prob, 4),
                "graphsage_probability": round(gnn_prob, 4),
                "tgn_probability": round(tgn_prob, 4),
                "ensemble_probability": ensemble_prob,
                "fraud_probability": ensemble_prob,
                "risk_classification": risk_class,
                "model_weights": weights,
            }
        ),
        200,
    )


@app.get("/model-info")
def model_info() -> tuple:
    """Return information about available models and their performance."""
    assets = _ensure_assets()
    
    info = {
        "models_available": {
            "baseline": "Random Forest",
            "graphsage": "Temporal GraphSAGE",
            "tgn": "Temporal Graph Network" if assets["tgn_model"] else "Not trained",
            "ensemble": "Weighted Ensemble",
        },
        "ensemble_weights": assets["ensemble_weights"],
        "risk_levels": {
            "critical": ">= 0.85",
            "high": "0.70 - 0.84",
            "medium": "0.45 - 0.69",
            "low": "0.25 - 0.44",
            "minimal": "< 0.25",
        },
        "features": {
            "advanced_node_features": [
                "pagerank", "cycle_participation", "flow_imbalance",
                "tx_velocity_per_hour", "burst_score", "dormancy_ratio",
                "near_threshold_ratio", "structuring_score",
                "tx_count_1h_max", "volume_24h_max",
            ],
        },
    }
    
    return jsonify(info), 200


if __name__ == "__main__":
    _ensure_assets()
    app.run(host="0.0.0.0", port=5000, debug=False)
