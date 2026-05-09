from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
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

from graph.temporal_graph import GraphBatch
from models.gnn import TemporalGraphSAGE
from models.tgn import TemporalGraphNetwork


ARTIFACT_ROOT = Path("artifacts")
MODELS_DIR = ARTIFACT_ROOT / "models"
PROCESSED_DIR = ARTIFACT_ROOT / "processed"
MAX_HISTORY_ROWS = 50000
HISTORY_EDGE_LIMIT = 500

app = Flask(__name__, static_folder="static")
CORS(app)
ASSETS = None


class PayloadValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _load_history_frame(preprocessors: dict, metadata: dict) -> pd.DataFrame:
    history_parts: list[pd.DataFrame] = []
    for name in ["train_transactions.csv", "test_transactions.csv"]:
        path = PROCESSED_DIR / name
        if path.exists():
            history_parts.append(pd.read_csv(path))

    edge_feature_names = metadata["edge_feature_names"]
    optional_columns = ["sender_gap_minutes", "receiver_gap_minutes"]
    required_columns = [
        "sender_id",
        "receiver_id",
        "transaction_amount",
        "transaction_type",
        "timestamp",
        *optional_columns,
        *edge_feature_names,
    ]

    if not history_parts:
        return pd.DataFrame(columns=required_columns)

    history = pd.concat(history_parts, ignore_index=True, sort=False)
    if "timestamp" not in history.columns:
        history["timestamp"] = pd.Timestamp.utcnow()
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    history = history.dropna(subset=["timestamp", "sender_id", "receiver_id"]).copy()

    for column in required_columns:
        if column not in history.columns:
            history[column] = 0.0

    # Ensure edge feature columns are present and numeric
    for feature in edge_feature_names:
        history[feature] = pd.to_numeric(history[feature], errors="coerce").fillna(0.0)

    history["transaction_amount"] = pd.to_numeric(history["transaction_amount"], errors="coerce").fillna(0.0)
    history["transaction_type"] = history["transaction_type"].astype(str).str.lower()

    return history[required_columns].sort_values("timestamp").reset_index(drop=True)


def _load_assets() -> dict:
    with open(PROCESSED_DIR / "preprocessors.pkl", "rb") as handle:
        preprocessors = pickle.load(handle)
    with open(MODELS_DIR / "gnn_node_embeddings.pkl", "rb") as handle:
        embedding_assets = pickle.load(handle)
    metadata = json.loads((PROCESSED_DIR / "feature_metadata.json").read_text())

    logistic_model = None
    if (MODELS_DIR / "logistic_regression.joblib").exists():
        logistic_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")

    baseline_model = joblib.load(MODELS_DIR / "random_forest.joblib")

    gnn_model = TemporalGraphSAGE(
        node_dim=embedding_assets["node_feature_dim"],
        edge_dim=embedding_assets["edge_feature_dim"],
        hidden_dim=64,
    )
    gnn_model.load_state_dict(torch.load(MODELS_DIR / "temporal_graphsage.pt", map_location="cpu"))
    gnn_model.eval()

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

    ensemble_weights = {"baseline": 0.2, "graphsage": 0.3, "tgn": 0.5}
    calibrator = None

    if (MODELS_DIR / "optimal_weights.json").exists():
        ensemble_weights = json.loads((MODELS_DIR / "optimal_weights.json").read_text())

    calibrator_info = {"enabled": False, "reason": "no_calibrator_file", "unique_levels": 0}

    if (MODELS_DIR / "calibrator.pkl").exists():
        with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)

        unique_levels = 0
        if hasattr(calibrator, "y_thresholds_"):
            unique_levels = int(len(np.unique(np.round(np.asarray(calibrator.y_thresholds_), 6))))

        if unique_levels <= 5:
            calibrator = None
            calibrator_info = {
                "enabled": False,
                "reason": "disabled_low_resolution_isotonic",
                "unique_levels": unique_levels,
            }
        else:
            calibrator_info = {
                "enabled": True,
                "reason": "enabled",
                "unique_levels": unique_levels,
            }

    calibration_enabled_env = os.getenv("AML_ENABLE_CALIBRATION", "0") == "1"
    if calibrator is not None and not calibration_enabled_env:
        calibrator = None
        calibrator_info = {
            "enabled": False,
            "reason": "disabled_by_default_set_AML_ENABLE_CALIBRATION_1_to_enable",
            "unique_levels": calibrator_info.get("unique_levels", 0),
        }

    node_frame: pd.DataFrame = preprocessors["node_frame"].copy()
    node_feature_names: list[str] = metadata["node_feature_names"]
    node_defaults = node_frame[node_feature_names].median(numeric_only=True).to_dict()

    history_frame = _load_history_frame(preprocessors, metadata)

    return {
        "preprocessors": preprocessors,
        "embeddings": embedding_assets,
        "metadata": metadata,
        "logistic_model": logistic_model,
        "baseline_model": baseline_model,
        "gnn_model": gnn_model,
        "tgn_model": tgn_model,
        "ensemble_weights": ensemble_weights,
        "calibrator": calibrator,
        "calibrator_info": calibrator_info,
        "node_feature_names": node_feature_names,
        "node_defaults": node_defaults,
        "node_frame_indexed": node_frame.set_index("account_id"),
        "history_frame": history_frame,
        "dynamic_account_to_index": dict(preprocessors["account_to_index"]),
        "dynamic_node_matrix": np.asarray(preprocessors["node_matrix"], dtype=np.float32).copy(),
        "base_account_set": set(preprocessors["account_to_index"].keys()),
    }


def _ensure_assets() -> dict:
    global ASSETS
    if ASSETS is None:
        ASSETS = _load_assets()
    return ASSETS


def _validation_error(message: str, details: dict | None = None, status_code: int = 400) -> tuple:
    payload = {"error": message}
    if details:
        payload.update(details)
    return jsonify(payload), status_code


def _normalize_transaction_payload(payload: dict, assets: dict) -> dict:
    if not isinstance(payload, dict):
        raise PayloadValidationError("Request body must be a JSON object.")

    required_fields = {"sender_id", "receiver_id", "transaction_amount", "timestamp", "transaction_type"}
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise PayloadValidationError(f"Missing fields: {', '.join(missing)}")

    try:
        amount = float(payload["transaction_amount"])
    except (TypeError, ValueError):
        raise PayloadValidationError("transaction_amount must be numeric") from None

    if amount < 0:
        raise PayloadValidationError("transaction_amount must be non-negative")

    transaction_type = str(payload["transaction_type"]).strip().lower()
    allowed_types = {str(value).lower() for value in assets["preprocessors"]["type_encoder"].classes_}
    if transaction_type not in allowed_types:
        raise PayloadValidationError(
            "Unsupported transaction_type.",
            details={"allowed_transaction_types": sorted(allowed_types)},
        )

    timestamp_value = str(payload["timestamp"])
    try:
        parsed_timestamp = pd.to_datetime(timestamp_value, utc=True).tz_convert(None)
        datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
    except ValueError:
        raise PayloadValidationError("timestamp must be ISO-8601 compatible") from None

    normalized = dict(payload)
    normalized["sender_id"] = str(payload["sender_id"]).strip()
    normalized["receiver_id"] = str(payload["receiver_id"]).strip()
    normalized["transaction_amount"] = amount
    normalized["transaction_type"] = transaction_type
    normalized["timestamp"] = parsed_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return normalized


def _build_context_history(recent_transactions: object, assets: dict) -> pd.DataFrame:
    if recent_transactions in (None, [], {}):
        return pd.DataFrame()

    if isinstance(recent_transactions, dict):
        for key in ("transactions", "history", "items", "context"):
            value = recent_transactions.get(key)
            if isinstance(value, list):
                recent_transactions = value
                break

    if not isinstance(recent_transactions, list):
        raise PayloadValidationError("recent_transactions must be an array of transaction objects")

    edge_feature_names = assets["metadata"]["edge_feature_names"]
    rows: list[dict] = []

    for index, item in enumerate(recent_transactions, start=1):
        if not isinstance(item, dict):
            raise PayloadValidationError(f"recent_transactions[{index}] must be an object")

        normalized = _normalize_transaction_payload(item, assets)
        row = {
            "sender_id": normalized["sender_id"],
            "receiver_id": normalized["receiver_id"],
            "transaction_amount": float(normalized["transaction_amount"]),
            "transaction_type": normalized["transaction_type"],
            "timestamp": pd.to_datetime(normalized["timestamp"], utc=True).tz_convert(None),
        }

        for feature in edge_feature_names:
            raw_value = item.get(feature, 0.0)
            try:
                row[feature] = float(raw_value)
            except (TypeError, ValueError):
                row[feature] = 0.0

        rows.append(row)

    history = pd.DataFrame(rows)
    if history.empty:
        return history

    for feature in edge_feature_names:
        if feature not in history.columns:
            history[feature] = 0.0
        history[feature] = pd.to_numeric(history[feature], errors="coerce").fillna(0.0)

    history["transaction_amount"] = pd.to_numeric(history["transaction_amount"], errors="coerce").fillna(0.0)
    history["transaction_type"] = history["transaction_type"].astype(str).str.lower()
    return history.sort_values("timestamp").reset_index(drop=True)


def _prepare_scoring_assets(assets: dict, history_override: pd.DataFrame | None = None, simulate_only: bool = False) -> dict:
    if history_override is None and not simulate_only:
        return assets

    scoped = dict(assets)
    history_frame = assets["history_frame"]
    if history_override is not None and not history_override.empty:
        history_frame = pd.concat([history_frame, history_override], ignore_index=True, sort=False)
        history_frame = history_frame.sort_values("timestamp").reset_index(drop=True)

    scoped["history_frame"] = history_frame.copy()
    scoped["dynamic_account_to_index"] = dict(assets["dynamic_account_to_index"])
    scoped["dynamic_node_matrix"] = np.asarray(assets["dynamic_node_matrix"], dtype=np.float32).copy()
    scoped["base_account_set"] = set(assets["base_account_set"])
    return scoped


def _context_summary(context_history: pd.DataFrame | None, payload: dict) -> dict:
    if context_history is None or context_history.empty:
        return {
            "context_applied": False,
            "recent_transactions": 0,
            "window_minutes": 0,
            "sender_recent_transactions": 0,
            "receiver_recent_transactions": 0,
        }

    timestamp = pd.to_datetime(payload["timestamp"], utc=True).tz_convert(None)
    window_minutes = float((timestamp - context_history["timestamp"].min()).total_seconds() / 60.0)
    sender_count = int((context_history["sender_id"] == payload["sender_id"]).sum())
    receiver_count = int((context_history["receiver_id"] == payload["receiver_id"]).sum())

    return {
        "context_applied": True,
        "recent_transactions": int(len(context_history)),
        "window_minutes": round(max(window_minutes, 0.0), 2),
        "context_start": context_history["timestamp"].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context_end": context_history["timestamp"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sender_recent_transactions": sender_count,
        "receiver_recent_transactions": receiver_count,
    }


def _score_transaction(
    raw_payload: dict,
    assets: dict,
    recent_transactions: object | None = None,
    simulate_only: bool = False,
) -> dict:
    payload = _normalize_transaction_payload(raw_payload, assets)
    context_history = _build_context_history(recent_transactions, assets)
    use_scoped_assets = simulate_only or not context_history.empty
    scoring_assets = _prepare_scoring_assets(assets, context_history, use_scoped_assets)

    (
        baseline_vector,
        node_features,
        edge_index,
        edge_attr,
        edge_timestamps,
        current_edge_id,
        engineered,
        inference_warnings,
    ) = _build_features(payload, scoring_assets)

    if not np.isfinite(baseline_vector).all():
        raise PayloadValidationError("Non-finite baseline features encountered", status_code=500)
    if not torch.isfinite(node_features).all() or not torch.isfinite(edge_attr).all():
        raise PayloadValidationError("Non-finite graph features encountered", status_code=500)

    logistic_prob = None
    if scoring_assets.get("logistic_model") is not None:
        logistic_prob = float(scoring_assets["logistic_model"].predict_proba(baseline_vector)[0, 1])

    baseline_prob = float(scoring_assets["baseline_model"].predict_proba(baseline_vector)[0, 1])

    with torch.no_grad():
        batch = GraphBatch(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            labels=torch.zeros(edge_index.size(1), dtype=torch.float32),
            edge_ids=torch.arange(edge_index.size(1), dtype=torch.long),
        )
        gnn_logits, _ = scoring_assets["gnn_model"](batch)
        gnn_prob = float(torch.sigmoid(gnn_logits[current_edge_id]).item())

    tgn_prob = gnn_prob
    if scoring_assets["tgn_model"] is not None:
        with torch.no_grad():
            tgn_logits, _, _ = scoring_assets["tgn_model"](
                node_features,
                edge_index,
                edge_attr,
                edge_timestamps=edge_timestamps,
            )
            tgn_prob = float(torch.sigmoid(tgn_logits[current_edge_id]).item())

    weights = scoring_assets["ensemble_weights"]
    raw_ensemble_prob = (
        float(weights["baseline"]) * baseline_prob
        + float(weights["graphsage"]) * gnn_prob
        + float(weights["tgn"]) * tgn_prob
    )

    ensemble_prob = raw_ensemble_prob
    calibration_applied = False
    if scoring_assets["calibrator"] is not None:
        ensemble_prob = float(scoring_assets["calibrator"].predict(np.array([ensemble_prob]))[0])
        calibration_applied = True

    ensemble_prob = float(np.clip(ensemble_prob, 0.0, 1.0))
    ml_only_prob = ensemble_prob

    ensemble_prob, aml_rules_triggered = _apply_advanced_aml_rules(
        payload=payload,
        engineered=engineered,
        ensemble_prob=ensemble_prob,
        assets=scoring_assets,
    )
    rules_applied = len(aml_rules_triggered) > 0

    risk_class = _risk_class(ensemble_prob)
    inference_warnings.extend(_extra_inference_warnings(payload, pd.to_datetime(payload["timestamp"], utc=True).tz_convert(None), float(payload["transaction_amount"])))

    for rule in aml_rules_triggered:
        inference_warnings.append(f"[AML Rule {rule['rule_id']}] {rule['rule_name']}: {rule['detail']}")

    explainability = _build_explainability_payload(
        payload=payload,
        engineered=engineered,
        baseline_vector=baseline_vector,
        baseline_prob=baseline_prob,
        gnn_prob=gnn_prob,
        tgn_prob=tgn_prob,
        weights=weights,
        pre_calibration_prob=float(np.clip(raw_ensemble_prob, 0.0, 1.0)),
        final_prob=ensemble_prob,
        risk_class=risk_class,
        inference_warnings=inference_warnings,
        assets=scoring_assets,
    )

    response = {
        "baseline_probability": round(baseline_prob, 4),
        "graphsage_probability": round(gnn_prob, 4),
        "tgn_probability": round(tgn_prob, 4),
        "ml_ensemble_probability": round(ml_only_prob, 4),
        "ensemble_probability": round(ensemble_prob, 4),
        "fraud_probability": round(ensemble_prob, 4),
        "risk_classification": risk_class,
        "model_weights": weights,
        "calibration_applied": calibration_applied,
        "calibrator_info": scoring_assets["calibrator_info"],
        "aml_rules_applied": rules_applied,
        "aml_rules_triggered": aml_rules_triggered,
        "aml_rules_count": len(aml_rules_triggered),
        "inference_warnings": inference_warnings,
        "explainability": explainability,
        "history_updated": not simulate_only,
        "context_applied": not context_history.empty,
        "context_summary": _context_summary(context_history, payload),
    }

    if logistic_prob is not None:
        response["logistic_probability"] = round(logistic_prob, 4)

    if not simulate_only:
        _append_to_history(payload, engineered, assets)

    return response


def _account_hash_factor(account_id: str) -> float:
    digest = hashlib.sha256(account_id.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:4], byteorder="big", signed=False) / float(2**32)
    return (unit - 0.5) * 0.3


def _compute_window_maxima(all_tx: pd.DataFrame, current_time: pd.Timestamp, minutes: int) -> tuple[int, float, int]:
    if all_tx.empty:
        return 0, 0.0, 0
    start = current_time - pd.Timedelta(minutes=minutes)
    win = all_tx[(all_tx["timestamp"] >= start) & (all_tx["timestamp"] <= current_time)]
    if win.empty:
        return 0, 0.0, 0
    counterparties = len(set(win["sender_id"]).union(set(win["receiver_id"]))) - 1
    return int(len(win)), float(win["transaction_amount"].sum()), max(counterparties, 0)


def _build_account_feature_row(account_id: str, current_time: pd.Timestamp, assets: dict) -> dict[str, float]:
    node_feature_names: list[str] = assets["node_feature_names"]
    defaults = assets["node_defaults"]

    base: dict[str, float] = {name: float(defaults.get(name, 0.0)) for name in node_feature_names}
    indexed = assets["node_frame_indexed"]

    if account_id in indexed.index:
        row = indexed.loc[account_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for name in node_feature_names:
            base[name] = float(row.get(name, base[name]))

    history = assets["history_frame"]
    scoped = history[history["timestamp"] <= current_time]
    outgoing = scoped[scoped["sender_id"] == account_id]
    incoming = scoped[scoped["receiver_id"] == account_id]

    out_degree = len(outgoing)
    in_degree = len(incoming)
    out_volume = float(outgoing["transaction_amount"].sum()) if out_degree else 0.0
    in_volume = float(incoming["transaction_amount"].sum()) if in_degree else 0.0

    all_tx = pd.concat([outgoing, incoming], ignore_index=True).sort_values("timestamp")

    base["out_degree"] = float(out_degree)
    base["in_degree"] = float(in_degree)
    base["out_volume"] = out_volume
    base["in_volume"] = in_volume
    base["avg_out_amount"] = float(outgoing["transaction_amount"].mean()) if out_degree else 0.0
    base["avg_in_amount"] = float(incoming["transaction_amount"].mean()) if in_degree else 0.0
    base["unique_receivers"] = float(outgoing["receiver_id"].nunique()) if out_degree else 0.0
    base["unique_senders"] = float(incoming["sender_id"].nunique()) if in_degree else 0.0

    denom = out_volume + in_volume + 1e-8
    base["flow_imbalance"] = float((out_volume - in_volume) / denom)

    if len(all_tx) >= 2:
        span_hours = max((all_tx["timestamp"].max() - all_tx["timestamp"].min()).total_seconds() / 3600.0, 1.0)
        base["tx_velocity_per_hour"] = float(len(all_tx) / span_hours)

        all_tx = all_tx.copy()
        all_tx["hour_bucket"] = all_tx["timestamp"].dt.floor("h")
        hourly_counts = all_tx.groupby("hour_bucket").size()
        max_hourly = int(hourly_counts.max())
        avg_hourly = float(hourly_counts.mean()) if len(hourly_counts) else 0.0
        base["max_hourly_count"] = float(max_hourly)
        base["burst_score"] = float(max_hourly / max(avg_hourly, 1.0))

        total_hours = max(int(span_hours), 1)
        active_hours = len(hourly_counts)
        base["dormancy_ratio"] = float(max(0.0, 1 - (active_hours / total_hours)))

        if len(hourly_counts) > 1:
            sorted_counts = np.sort(hourly_counts.values)
            cumsum = np.cumsum(sorted_counts)
            n = len(sorted_counts)
            concentration = (n + 1 - 2 * np.sum(cumsum) / max(cumsum[-1], 1e-8)) / n
            base["activity_concentration"] = float(max(0.0, concentration))
        else:
            base["activity_concentration"] = 0.0
    else:
        base["tx_velocity_per_hour"] = 0.0
        base["max_hourly_count"] = 0.0
        base["burst_score"] = 0.0
        base["dormancy_ratio"] = 1.0
        base["activity_concentration"] = 0.0

    amounts = outgoing["transaction_amount"].to_numpy(dtype=np.float64) if out_degree else np.array([])
    if amounts.size:
        near_threshold = float(np.mean((amounts >= 9000.0) & (amounts < 10000.0)))
        round_ratio = float(np.mean(np.mod(amounts, 100.0) < 1e-8))
        cv = float(np.std(amounts) / (np.mean(amounts) + 1e-8)) if amounts.size > 1 else 1.0
        variance_ratio = float(1.0 / (1.0 + cv))
        structuring_score = float(0.5 * near_threshold + 0.3 * round_ratio + 0.2 * variance_ratio)
    else:
        near_threshold = 0.0
        round_ratio = 0.0
        variance_ratio = 0.0
        structuring_score = 0.0

    base["near_threshold_ratio"] = near_threshold
    base["round_number_ratio"] = round_ratio
    base["amount_variance_ratio"] = variance_ratio
    base["structuring_score"] = structuring_score

    for window_name, minutes in [("1h", 60), ("6h", 360), ("24h", 1440)]:
        count, volume, counterparties = _compute_window_maxima(all_tx, current_time, minutes)
        base[f"tx_count_{window_name}_max"] = float(count)
        base[f"volume_{window_name}_max"] = float(volume)
        base[f"unique_counterparties_{window_name}_max"] = float(counterparties)

    if account_id not in assets["base_account_set"]:
        # Deterministic variation avoids unknown-account collapse to one identical vector.
        factor = _account_hash_factor(account_id)
        base["flow_imbalance"] = float(np.clip(base.get("flow_imbalance", 0.0) + factor, -1.0, 1.0))
        base["activity_concentration"] = float(
            np.clip(base.get("activity_concentration", 0.0) + abs(factor), 0.0, 1.0)
        )
        base["pagerank"] = float(max(base.get("pagerank", 0.0) * (1.0 + factor), 0.0))

    return {name: float(base.get(name, 0.0)) for name in node_feature_names}


def _compute_gap_minutes(
    account_id: str,
    current_time: pd.Timestamp,
    history: pd.DataFrame,
    role_column: str,
    fallback_gap_column: str,
) -> float:
    scoped = history[(history[role_column] == account_id) & (history["timestamp"] < current_time)]
    if scoped.empty:
        if fallback_gap_column not in history.columns:
            return 0.0

        fallback = pd.to_numeric(history[fallback_gap_column], errors="coerce").median()
        if pd.isna(fallback):
            return 0.0
        return float(max(fallback, 0.0))

    latest_ts = scoped["timestamp"].max()
    return float(max((current_time - latest_ts).total_seconds() / 60.0, 0.0))


def _upsert_dynamic_node(account_id: str, raw_row: dict[str, float], assets: dict) -> int:
    node_feature_names = assets["node_feature_names"]
    node_scaler = assets["preprocessors"]["node_scaler"]
    account_to_index = assets["dynamic_account_to_index"]
    node_matrix = assets["dynamic_node_matrix"]

    row_df = pd.DataFrame([[raw_row[name] for name in node_feature_names]], columns=node_feature_names)
    scaled = node_scaler.transform(row_df).astype(np.float32)[0]

    if account_id in account_to_index:
        idx = int(account_to_index[account_id])
        node_matrix[idx] = scaled
        assets["dynamic_node_matrix"] = node_matrix
        return idx

    idx = int(node_matrix.shape[0])
    account_to_index[account_id] = idx
    assets["dynamic_node_matrix"] = np.vstack([node_matrix, scaled])
    return idx


def _encode_edge_features(
    payload: dict,
    timestamp: pd.Timestamp,
    sender_gap_minutes: float,
    receiver_gap_minutes: float,
    assets: dict,
) -> tuple[dict[str, float], np.ndarray]:
    preprocessors = assets["preprocessors"]

    amount = float(payload["transaction_amount"])
    tx_type = str(payload["transaction_type"]).strip().lower()
    tx_type_encoded = int(preprocessors["type_encoder"].transform([tx_type])[0])

    amount_scaled = float(
        preprocessors["amount_scaler"].transform(pd.DataFrame({"transaction_amount": [amount]}))[0, 0]
    )

    gap_scaled = preprocessors["gap_scaler"].transform(
        pd.DataFrame(
            {
                "sender_gap_minutes": [sender_gap_minutes],
                "receiver_gap_minutes": [receiver_gap_minutes],
            }
        )
    )[0]

    hour = int(timestamp.hour)
    day_of_week = int(timestamp.dayofweek)
    is_weekend = int(day_of_week in [5, 6])

    engineered = {
        "transaction_amount": amount,
        "transaction_amount_scaled": amount_scaled,
        "transaction_type": tx_type,
        "transaction_type_encoded": tx_type_encoded,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "is_weekend": is_weekend,
        "sender_gap_minutes": sender_gap_minutes,
        "receiver_gap_minutes": receiver_gap_minutes,
        "sender_gap_minutes_scaled": float(gap_scaled[0]),
        "receiver_gap_minutes_scaled": float(gap_scaled[1]),
    }

    edge_vector = np.array(
        [
            engineered["transaction_amount_scaled"],
            float(engineered["transaction_type_encoded"]),
            engineered["hour_sin"],
            engineered["hour_cos"],
            engineered["dow_sin"],
            engineered["dow_cos"],
            float(engineered["is_weekend"]),
            engineered["sender_gap_minutes_scaled"],
            engineered["receiver_gap_minutes_scaled"],
        ],
        dtype=np.float32,
    )

    return engineered, edge_vector


def _build_features(payload: dict, assets: dict) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, dict, list[str]]:
    history = assets["history_frame"]
    timestamp = pd.to_datetime(payload["timestamp"], utc=True).tz_convert(None)

    sender_id = str(payload["sender_id"])
    receiver_id = str(payload["receiver_id"])

    sender_row = _build_account_feature_row(sender_id, timestamp, assets)
    receiver_row = _build_account_feature_row(receiver_id, timestamp, assets)

    sender_gap_minutes = _compute_gap_minutes(
        sender_id,
        timestamp,
        history,
        role_column="sender_id",
        fallback_gap_column="sender_gap_minutes",
    )
    receiver_gap_minutes = _compute_gap_minutes(
        receiver_id,
        timestamp,
        history,
        role_column="receiver_id",
        fallback_gap_column="receiver_gap_minutes",
    )

    engineered, edge_vector = _encode_edge_features(
        payload,
        timestamp,
        sender_gap_minutes,
        receiver_gap_minutes,
        assets,
    )

    sender_features = {f"sender_{name}": value for name, value in sender_row.items()}
    receiver_features = {f"receiver_{name}": value for name, value in receiver_row.items()}

    features_for_baseline = {
        "transaction_amount_scaled": engineered["transaction_amount_scaled"],
        "transaction_type_encoded": engineered["transaction_type_encoded"],
        "hour_sin": engineered["hour_sin"],
        "hour_cos": engineered["hour_cos"],
        "dow_sin": engineered["dow_sin"],
        "dow_cos": engineered["dow_cos"],
        "is_weekend": engineered["is_weekend"],
        "sender_gap_minutes_scaled": engineered["sender_gap_minutes_scaled"],
        "receiver_gap_minutes_scaled": engineered["receiver_gap_minutes_scaled"],
        **sender_features,
        **receiver_features,
    }

    ordered_features = assets["metadata"]["transaction_feature_names"]
    baseline_vector = np.array([[features_for_baseline.get(name, 0.0) for name in ordered_features]], dtype=np.float32)

    sender_idx = _upsert_dynamic_node(sender_id, sender_row, assets)
    receiver_idx = _upsert_dynamic_node(receiver_id, receiver_row, assets)

    context = history[history["timestamp"] <= timestamp].tail(HISTORY_EDGE_LIMIT).copy()
    edge_feature_names = assets["metadata"]["edge_feature_names"]

    context_src: list[int] = []
    context_dst: list[int] = []
    context_edge_attr: list[np.ndarray] = []
    context_times: list[pd.Timestamp] = []

    dynamic_map = assets["dynamic_account_to_index"]

    for _, row in context.iterrows():
        src_id = str(row["sender_id"])
        dst_id = str(row["receiver_id"])

        if src_id not in dynamic_map:
            src = _upsert_dynamic_node(src_id, _build_account_feature_row(src_id, timestamp, assets), assets)
        else:
            src = int(dynamic_map[src_id])

        if dst_id not in dynamic_map:
            dst = _upsert_dynamic_node(dst_id, _build_account_feature_row(dst_id, timestamp, assets), assets)
        else:
            dst = int(dynamic_map[dst_id])

        context_src.append(src)
        context_dst.append(dst)
        context_edge_attr.append(np.array([float(row.get(name, 0.0)) for name in edge_feature_names], dtype=np.float32))
        context_times.append(pd.to_datetime(row["timestamp"]))

    context_src.append(sender_idx)
    context_dst.append(receiver_idx)
    context_edge_attr.append(edge_vector)
    context_times.append(timestamp)

    edge_index = torch.tensor(np.vstack([context_src, context_dst]), dtype=torch.long)
    edge_attr = torch.tensor(np.vstack(context_edge_attr), dtype=torch.float32)

    min_time = min(context_times)
    edge_timestamps = torch.tensor(
        [max((ts - min_time).total_seconds() / 60.0, 0.0) for ts in context_times],
        dtype=torch.float32,
    )

    node_features = torch.tensor(assets["dynamic_node_matrix"], dtype=torch.float32)
    current_edge_id = len(context_src) - 1

    warnings: list[str] = []
    if sender_id not in assets["base_account_set"]:
        warnings.append("sender_id unseen in training accounts; using adaptive fallback features")
    if receiver_id not in assets["base_account_set"]:
        warnings.append("receiver_id unseen in training accounts; using adaptive fallback features")

    return baseline_vector, node_features, edge_index, edge_attr, edge_timestamps, current_edge_id, engineered, warnings


def _append_to_history(payload: dict, engineered: dict, assets: dict) -> None:
    row = {
        "sender_id": str(payload["sender_id"]),
        "receiver_id": str(payload["receiver_id"]),
        "transaction_amount": float(payload["transaction_amount"]),
        "transaction_type": str(payload["transaction_type"]).strip().lower(),
        "timestamp": pd.to_datetime(payload["timestamp"], utc=True).tz_convert(None),
        "transaction_amount_scaled": engineered["transaction_amount_scaled"],
        "transaction_type_encoded": engineered["transaction_type_encoded"],
        "hour_sin": engineered["hour_sin"],
        "hour_cos": engineered["hour_cos"],
        "dow_sin": engineered["dow_sin"],
        "dow_cos": engineered["dow_cos"],
        "is_weekend": engineered["is_weekend"],
        "sender_gap_minutes_scaled": engineered["sender_gap_minutes_scaled"],
        "receiver_gap_minutes_scaled": engineered["receiver_gap_minutes_scaled"],
        "sender_gap_minutes": engineered["sender_gap_minutes"],
        "receiver_gap_minutes": engineered["receiver_gap_minutes"],
    }

    history = assets["history_frame"]
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    history = history.sort_values("timestamp").tail(MAX_HISTORY_ROWS).reset_index(drop=True)
    assets["history_frame"] = history


def _risk_class(probability: float) -> str:
    if probability >= 0.85:
        return "critical"
    if probability >= 0.70:
        return "high"
    if probability >= 0.45:
        return "medium"
    if probability >= 0.25:
        return "low"
    return "minimal"


def _apply_advanced_aml_rules(
    payload: dict,
    engineered: dict,
    ensemble_prob: float,
    assets: dict,
) -> tuple[float, list[dict]]:
    """Advanced rule-based AML heuristic engine.

    Evaluates the transaction against known money-laundering red flags and
    produces a rule-adjusted probability.  Each triggered rule contributes a
    boost that is combined with the ML ensemble score via a *max-override*
    strategy, so that clearly suspicious patterns are never under-scored.

    Returns the adjusted probability and a list of triggered rule dicts.
    """
    amount = float(payload["transaction_amount"])
    tx_type = str(payload["transaction_type"]).strip().lower()
    sender_id = str(payload["sender_id"])
    receiver_id = str(payload["receiver_id"])
    sender_gap = float(engineered.get("sender_gap_minutes", 999.0))
    receiver_gap = float(engineered.get("receiver_gap_minutes", 999.0))

    triggered_rules: list[dict] = []
    rule_prob = ensemble_prob  # start from ML score

    # ── Rule 1: High-value transaction thresholds ──────────────────────────
    if amount >= 50000:
        triggered_rules.append({
            "rule_id": "HV-3",
            "rule_name": "Extremely High Value Transaction",
            "severity": "critical",
            "min_probability": 0.92,
            "detail": f"Transaction amount ${amount:,.2f} exceeds $50,000 — extreme AML threshold.",
        })
    elif amount >= 10000:
        triggered_rules.append({
            "rule_id": "HV-2",
            "rule_name": "High Value Transaction",
            "severity": "critical",
            "min_probability": 0.88,
            "detail": f"Transaction amount ${amount:,.2f} meets or exceeds $10,000 — standard AML reporting threshold.",
        })
    elif amount >= 5000:
        triggered_rules.append({
            "rule_id": "HV-1",
            "rule_name": "Elevated Value Transaction",
            "severity": "high",
            "min_probability": 0.75,
            "detail": f"Transaction amount ${amount:,.2f} exceeds $5,000 — elevated monitoring threshold.",
        })
    elif amount >= 3000:
        triggered_rules.append({
            "rule_id": "HV-0",
            "rule_name": "Moderate Value Transaction",
            "severity": "medium",
            "min_probability": 0.55,
            "detail": f"Transaction amount ${amount:,.2f} exceeds $3,000 — moderate monitoring threshold.",
        })

    # ── Rule 2: Structuring / Smurfing detection ───────────────────────────
    is_near_threshold = 9000.0 <= amount < 10000.0
    is_round_number = abs(amount % 1000) < 1.0 and amount >= 1000.0
    is_just_under = any(abs(amount - t) <= (t * 0.05) and amount < t for t in [5000, 10000, 50000])

    if is_near_threshold:
        triggered_rules.append({
            "rule_id": "ST-1",
            "rule_name": "Structuring — Near Reporting Threshold",
            "severity": "critical",
            "min_probability": 0.90,
            "detail": f"Amount ${amount:,.2f} is just below $10,000 reporting threshold — classic structuring pattern.",
        })
    elif is_just_under:
        triggered_rules.append({
            "rule_id": "ST-2",
            "rule_name": "Structuring — Just Under Threshold",
            "severity": "high",
            "min_probability": 0.78,
            "detail": f"Amount ${amount:,.2f} is just below a key AML threshold — possible structuring.",
        })
    if is_round_number and amount >= 5000:
        triggered_rules.append({
            "rule_id": "ST-3",
            "rule_name": "Suspicious Round Amount",
            "severity": "high",
            "min_probability": 0.72,
            "detail": f"Large round-number transfer of ${amount:,.2f} — often associated with layering.",
        })

    # ── Rule 3: Rapid-fire / High velocity ────────────────────────────────
    if sender_gap < 2.0 and amount >= 1000:
        triggered_rules.append({
            "rule_id": "VEL-1",
            "rule_name": "Rapid Sender Activity",
            "severity": "high",
            "min_probability": 0.80,
            "detail": f"Sender transacted again within {sender_gap:.1f} minutes with ${amount:,.2f} — burst transfer behavior.",
        })
    elif sender_gap < 10.0 and amount >= 3000:
        triggered_rules.append({
            "rule_id": "VEL-2",
            "rule_name": "High Velocity Sender",
            "severity": "medium",
            "min_probability": 0.65,
            "detail": f"Sender transacted within {sender_gap:.1f} minutes with ${amount:,.2f} — elevated velocity.",
        })

    if receiver_gap < 2.0 and amount >= 1000:
        triggered_rules.append({
            "rule_id": "VEL-3",
            "rule_name": "Rapid Receiver Activity",
            "severity": "high",
            "min_probability": 0.78,
            "detail": f"Receiver received again within {receiver_gap:.1f} minutes — possible layering destination.",
        })

    # ── Rule 4: Self-transfer detection ────────────────────────────────────
    if sender_id == receiver_id:
        if amount >= 5000:
            triggered_rules.append({
                "rule_id": "SELF-1",
                "rule_name": "High-Value Self Transfer",
                "severity": "critical",
                "min_probability": 0.90,
                "detail": f"Self-transfer of ${amount:,.2f} between same account — strong laundering indicator.",
            })
        elif amount >= 1000:
            triggered_rules.append({
                "rule_id": "SELF-2",
                "rule_name": "Self Transfer",
                "severity": "high",
                "min_probability": 0.72,
                "detail": f"Self-transfer of ${amount:,.2f} — unusual account behavior.",
            })

    # ── Rule 5: Suspicious transaction types with high amounts ─────────────
    high_risk_types = {"withdrawal", "wire", "international", "crypto", "cash"}
    if tx_type in high_risk_types and amount >= 5000:
        triggered_rules.append({
            "rule_id": "TYPE-1",
            "rule_name": f"High-Risk Transaction Type: {tx_type.title()}",
            "severity": "high",
            "min_probability": 0.78,
            "detail": f"High-value {tx_type} of ${amount:,.2f} — this transaction type carries elevated AML risk.",
        })
    elif tx_type in high_risk_types and amount >= 2000:
        triggered_rules.append({
            "rule_id": "TYPE-2",
            "rule_name": f"Monitored Transaction Type: {tx_type.title()}",
            "severity": "medium",
            "min_probability": 0.55,
            "detail": f"{tx_type.title()} of ${amount:,.2f} — transaction type under enhanced monitoring.",
        })

    # ── Rule 6: Combined risk escalation ───────────────────────────────────
    severity_counts = {"critical": 0, "high": 0, "medium": 0}
    for rule in triggered_rules:
        sev = rule["severity"]
        if sev in severity_counts:
            severity_counts[sev] += 1

    if severity_counts["critical"] >= 2 or (severity_counts["critical"] >= 1 and severity_counts["high"] >= 1):
        triggered_rules.append({
            "rule_id": "ESC-1",
            "rule_name": "Multi-Rule Escalation",
            "severity": "critical",
            "min_probability": 0.95,
            "detail": f"Multiple high-severity AML rules triggered simultaneously — strongly suspicious transaction.",
        })
    elif severity_counts["high"] >= 2:
        triggered_rules.append({
            "rule_id": "ESC-2",
            "rule_name": "Combined Risk Escalation",
            "severity": "high",
            "min_probability": 0.82,
            "detail": f"Multiple AML risk indicators detected — combined risk escalation applied.",
        })

    # ── Apply max-override: rule floor beats ML if higher ──────────────────
    if triggered_rules:
        max_rule_prob = max(r["min_probability"] for r in triggered_rules)
        rule_prob = max(ensemble_prob, max_rule_prob)

    return float(np.clip(rule_prob, 0.0, 1.0)), triggered_rules


def _format_feature_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _feature_explanation(name: str, value: float) -> str:
    if name == "transaction_amount_scaled":
        return "Scaled transaction amount. Larger values can indicate elevated laundering risk."
    if name == "sender_gap_minutes_scaled":
        return "Time since sender's last activity. Very short gaps can indicate burst behavior."
    if name == "receiver_gap_minutes_scaled":
        return "Time since receiver's last activity. Unusual inactivity-to-activity jumps are monitored."
    if "structuring_score" in name:
        return "Structuring behavior score based on near-threshold and round-number transfer patterns."
    if "burst_score" in name:
        return "Burst intensity of account activity over short windows."
    if "velocity" in name:
        return "Transaction pace over time. Sudden acceleration may be suspicious."
    if "flow_imbalance" in name:
        return "Difference between incoming and outgoing value flows for an account."
    if "pagerank" in name:
        return "Graph centrality signal indicating influence in the transfer network."
    return "Model feature considered during risk scoring."


def _extract_feature_importances(model: object, feature_names: list[str]) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"), dtype=np.float64)
        if values.shape[0] == len(feature_names):
            return values

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in reversed(list(named_steps.values())):
            if hasattr(step, "feature_importances_"):
                values = np.asarray(getattr(step, "feature_importances_"), dtype=np.float64)
                if values.shape[0] == len(feature_names):
                    return values

    return np.ones(len(feature_names), dtype=np.float64) / max(len(feature_names), 1)


def _top_feature_factors(baseline_vector: np.ndarray, assets: dict, top_k: int = 6) -> list[dict]:
    names: list[str] = assets["metadata"]["transaction_feature_names"]
    importances = _extract_feature_importances(assets["baseline_model"], names)
    row = baseline_vector[0]
    scores = np.abs(row.astype(np.float64) * importances)
    ranked_idx = np.argsort(scores)[::-1][:top_k]

    factors: list[dict] = []
    for idx in ranked_idx:
        raw_value = float(row[idx])
        factors.append(
            {
                "feature": names[idx],
                "label": _format_feature_name(names[idx]),
                "value": round(raw_value, 6),
                "importance": round(float(importances[idx]), 6),
                "impact_score": round(float(scores[idx]), 6),
                "explanation": _feature_explanation(names[idx], raw_value),
            }
        )
    return factors


def _model_contributions(
    baseline_prob: float,
    gnn_prob: float,
    tgn_prob: float,
    weights: dict,
    pre_calibration_prob: float,
    post_calibration_prob: float,
) -> dict:
    contributions = {
        "baseline": float(weights["baseline"]) * baseline_prob,
        "graphsage": float(weights["graphsage"]) * gnn_prob,
        "tgn": float(weights["tgn"]) * tgn_prob,
    }

    total = max(sum(contributions.values()), 1e-9)
    share_pct = {key: round(100.0 * value / total, 2) for key, value in contributions.items()}

    return {
        "per_model": {
            "baseline": {
                "probability": round(baseline_prob, 4),
                "weighted_contribution": round(contributions["baseline"], 6),
                "contribution_percent": share_pct["baseline"],
            },
            "graphsage": {
                "probability": round(gnn_prob, 4),
                "weighted_contribution": round(contributions["graphsage"], 6),
                "contribution_percent": share_pct["graphsage"],
            },
            "tgn": {
                "probability": round(tgn_prob, 4),
                "weighted_contribution": round(contributions["tgn"], 6),
                "contribution_percent": share_pct["tgn"],
            },
        },
        "pre_calibration_probability": round(pre_calibration_prob, 4),
        "post_calibration_probability": round(post_calibration_prob, 4),
        "calibration_adjustment": round(post_calibration_prob - pre_calibration_prob, 6),
    }


def _confidence_summary(
    baseline_prob: float,
    gnn_prob: float,
    tgn_prob: float,
    inference_warnings: list[str],
) -> dict:
    probs = np.array([baseline_prob, gnn_prob, tgn_prob], dtype=np.float64)
    spread = float(np.max(probs) - np.min(probs))
    agreement = float(1.0 - np.clip(spread, 0.0, 1.0))

    warning_penalty = min(len(inference_warnings) * 0.12, 0.4)
    confidence_score = float(np.clip(agreement - warning_penalty, 0.0, 1.0))

    if confidence_score >= 0.75:
        label = "high"
    elif confidence_score >= 0.45:
        label = "moderate"
    else:
        label = "low"

    return {
        "score": round(confidence_score, 4),
        "label": label,
        "model_agreement": round(agreement, 4),
        "model_spread": round(spread, 4),
    }


def _plain_language_summary(
    ensemble_prob: float,
    risk_class: str,
    top_factors: list[dict],
    confidence: dict,
) -> dict:
    percent = round(ensemble_prob * 100.0, 1)
    factor_text = ", ".join([factor["label"].lower() for factor in top_factors[:3]])

    if risk_class in {"critical", "high"}:
        headline = f"High money-laundering risk detected ({percent}%)."
        action = "Escalate for immediate compliance review and enhanced due diligence."
    elif risk_class == "medium":
        headline = f"Moderate money-laundering risk detected ({percent}%)."
        action = "Queue for analyst validation with supporting account history checks."
    else:
        headline = f"Low money-laundering risk detected ({percent}%)."
        action = "No immediate escalation needed; continue routine monitoring."

    confidence_note = f"Model confidence is {confidence['label']} (score {round(confidence['score'] * 100, 1)}%)."

    return {
        "headline": headline,
        "drivers": f"Top contributing feature areas: {factor_text}.",
        "confidence_note": confidence_note,
        "recommended_action": action,
    }


def _build_decision_steps(
    payload: dict,
    engineered: dict,
    contributions: dict,
    risk_class: str,
    confidence: dict,
) -> list[dict]:
    amount = float(payload["transaction_amount"])
    sender_gap = float(engineered.get("sender_gap_minutes", 0.0))
    receiver_gap = float(engineered.get("receiver_gap_minutes", 0.0))

    return [
        {
            "step": 1,
            "title": "Transaction context normalized",
            "detail": f"Amount {amount:.2f} with type '{payload['transaction_type']}' was encoded for model input.",
        },
        {
            "step": 2,
            "title": "Temporal behavior analyzed",
            "detail": f"Sender gap {sender_gap:.1f} min and receiver gap {receiver_gap:.1f} min were compared to historical activity.",
        },
        {
            "step": 3,
            "title": "Model signals combined",
            "detail": (
                "Weighted model impacts were baseline "
                f"{contributions['per_model']['baseline']['contribution_percent']}%, "
                f"graphsage {contributions['per_model']['graphsage']['contribution_percent']}%, "
                f"tgn {contributions['per_model']['tgn']['contribution_percent']}%."
            ),
        },
        {
            "step": 4,
            "title": "Risk tier assigned",
            "detail": f"Final risk classified as '{risk_class}' with {confidence['label']} confidence.",
        },
    ]


def _extra_inference_warnings(
    payload: dict,
    timestamp: pd.Timestamp,
    amount: float,
) -> list[str]:
    warnings: list[str] = []
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)

    age_days = (now - timestamp).total_seconds() / (24 * 3600)
    if age_days > 365:
        warnings.append("timestamp older than one year; behavior context may be stale")
    if age_days < -1:
        warnings.append("timestamp is in the future; verify input clock synchronization")
    if amount >= 50000:
        warnings.append("transaction amount is very high; additional due diligence recommended")
    if str(payload["sender_id"]) == str(payload["receiver_id"]):
        warnings.append("sender and receiver are identical; self-transfer pattern detected")
    return warnings


def _build_explainability_payload(
    payload: dict,
    engineered: dict,
    baseline_vector: np.ndarray,
    baseline_prob: float,
    gnn_prob: float,
    tgn_prob: float,
    weights: dict,
    pre_calibration_prob: float,
    final_prob: float,
    risk_class: str,
    inference_warnings: list[str],
    assets: dict,
) -> dict:
    top_factors = _top_feature_factors(baseline_vector, assets, top_k=6)
    contributions = _model_contributions(
        baseline_prob=baseline_prob,
        gnn_prob=gnn_prob,
        tgn_prob=tgn_prob,
        weights=weights,
        pre_calibration_prob=pre_calibration_prob,
        post_calibration_prob=final_prob,
    )
    confidence = _confidence_summary(
        baseline_prob=baseline_prob,
        gnn_prob=gnn_prob,
        tgn_prob=tgn_prob,
        inference_warnings=inference_warnings,
    )
    summary = _plain_language_summary(
        ensemble_prob=final_prob,
        risk_class=risk_class,
        top_factors=top_factors,
        confidence=confidence,
    )
    steps = _build_decision_steps(
        payload=payload,
        engineered=engineered,
        contributions=contributions,
        risk_class=risk_class,
        confidence=confidence,
    )

    return {
        "summary": summary,
        "confidence": confidence,
        "model_contributions": contributions,
        "top_factors": top_factors,
        "decision_steps": steps,
        "inputs": {
            "sender_id": str(payload["sender_id"]),
            "receiver_id": str(payload["receiver_id"]),
            "transaction_type": str(payload["transaction_type"]),
            "transaction_amount": round(float(payload["transaction_amount"]), 2),
            "sender_gap_minutes": round(float(engineered.get("sender_gap_minutes", 0.0)), 3),
            "receiver_gap_minutes": round(float(engineered.get("receiver_gap_minutes", 0.0)), 3),
        },
    }


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
    recent_transactions = payload.get("recent_transactions") if isinstance(payload, dict) else None

    try:
        response = _score_transaction(
            raw_payload=payload,
            assets=assets,
            recent_transactions=recent_transactions,
            simulate_only=bool(payload.get("simulate_only", False)) if isinstance(payload, dict) else False,
        )
    except PayloadValidationError as exc:
        return _validation_error(exc.message, exc.details, exc.status_code)

    return jsonify(response), 200


@app.post("/predict-context")
def predict_context() -> tuple:
    assets = _ensure_assets()
    body = request.get_json(force=True)

    if not isinstance(body, dict):
        return _validation_error("Request body must be a JSON object.")

    payload = body.get("transaction") if isinstance(body.get("transaction"), dict) else body
    recent_transactions = body.get("recent_transactions") or body.get("history") or []

    try:
        response = _score_transaction(
            raw_payload=payload,
            assets=assets,
            recent_transactions=recent_transactions,
            simulate_only=bool(body.get("simulate_only", True)),
        )
        response["context_mode"] = True
    except PayloadValidationError as exc:
        return _validation_error(exc.message, exc.details, exc.status_code)

    return jsonify(response), 200


@app.post("/batch-predict")
def batch_predict() -> tuple:
    assets = _ensure_assets()
    body = request.get_json(force=True)

    if isinstance(body, list):
        cases = body
    elif isinstance(body, dict):
        cases = body.get("cases") or body.get("transactions") or body.get("items")
    else:
        cases = None

    if not isinstance(cases, list) or not cases:
        return _validation_error("cases must be a non-empty array")

    results: list[dict] = []
    successful_results: list[dict] = []

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            results.append(
                {
                    "case_id": f"case_{index:02d}",
                    "status": "error",
                    "error": "Case payload must be an object.",
                }
            )
            continue

        raw_payload = case.get("payload") or case.get("transaction") or case
        recent_transactions = case.get("recent_transactions") or case.get("history")
        case_id = str(case.get("case_id") or case.get("id") or f"case_{index:02d}")
        scenario = str(case.get("scenario") or case.get("name") or case_id)

        try:
            response = _score_transaction(
                raw_payload=raw_payload,
                assets=assets,
                recent_transactions=recent_transactions,
                simulate_only=True,
            )
            response.update(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "status": "ok",
                }
            )
            results.append(response)
            successful_results.append(response)
        except PayloadValidationError as exc:
            results.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "status": "error",
                    "error": exc.message,
                    "details": exc.details,
                }
            )

    ensemble_values = [float(row["ensemble_probability"]) for row in successful_results]
    risk_counts = {}
    for row in successful_results:
        risk_class = str(row["risk_classification"])
        risk_counts[risk_class] = risk_counts.get(risk_class, 0) + 1

    summary = {
        "cases_requested": len(cases),
        "cases_completed": len(successful_results),
        "cases_failed": len(results) - len(successful_results),
        "unique_probability_count": len({round(value, 4) for value in ensemble_values}),
        "average_probability": round(float(np.mean(ensemble_values)), 4) if ensemble_values else 0.0,
        "min_probability": round(float(np.min(ensemble_values)), 4) if ensemble_values else 0.0,
        "max_probability": round(float(np.max(ensemble_values)), 4) if ensemble_values else 0.0,
        "risk_counts": risk_counts,
    }

    return (
        jsonify(
            {
                "results": results,
                "summary": summary,
                "batch_mode": True,
            }
        ),
        200,
    )


@app.get("/model-info")
def model_info() -> tuple:
    assets = _ensure_assets()

    info = {
        "models_available": {
            "logistic_regression": "Logistic Regression" if assets.get("logistic_model") is not None else "Not loaded",
            "baseline": "Random Forest",
            "graphsage": "Temporal GraphSAGE",
            "tgn": "Temporal Graph Network" if assets["tgn_model"] else "Not trained",
            "ensemble": "Weighted Ensemble",
        },
        "ensemble_weights": assets["ensemble_weights"],
        "calibrator": assets["calibrator_info"],
        "risk_levels": {
            "critical": ">= 0.85",
            "high": "0.70 - 0.84",
            "medium": "0.45 - 0.69",
            "low": "0.25 - 0.44",
            "minimal": "< 0.25",
        },
        "features": {
            "advanced_node_features": [
                "pagerank",
                "cycle_participation",
                "flow_imbalance",
                "tx_velocity_per_hour",
                "burst_score",
                "dormancy_ratio",
                "near_threshold_ratio",
                "structuring_score",
                "tx_count_1h_max",
                "volume_24h_max",
            ],
            "tooltip_hints": {
                "transaction_amount_scaled": _feature_explanation("transaction_amount_scaled", 0.0),
                "sender_gap_minutes_scaled": _feature_explanation("sender_gap_minutes_scaled", 0.0),
                "receiver_gap_minutes_scaled": _feature_explanation("receiver_gap_minutes_scaled", 0.0),
                "sender_structuring_score": _feature_explanation("sender_structuring_score", 0.0),
                "sender_burst_score": _feature_explanation("sender_burst_score", 0.0),
                "sender_flow_imbalance": _feature_explanation("sender_flow_imbalance", 0.0),
            },
        },
    }

    return jsonify(info), 200


if __name__ == "__main__":
    _ensure_assets()
    app.run(host="0.0.0.0", port=5000, debug=False)
