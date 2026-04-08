from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from graph.temporal_graph import GraphBatch


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_graph: GraphBatch
    test_graph: GraphBatch
    node_frame: pd.DataFrame
    account_to_index: dict[str, int]
    transaction_feature_names: list[str]
    node_feature_names: list[str]
    temporal_sequences: dict  # For TGN temporal batching


def _compute_pagerank(df: pd.DataFrame, accounts: list[str], damping: float = 0.85, iterations: int = 20) -> dict[str, float]:
    """Compute PageRank scores for account graph."""
    n = len(accounts)
    account_idx = {acc: i for i, acc in enumerate(accounts)}
    
    # Build adjacency with edge weights (transaction counts)
    out_edges = defaultdict(list)
    edge_weights = defaultdict(float)
    for _, row in df.iterrows():
        src, dst = row["sender_id"], row["receiver_id"]
        out_edges[src].append(dst)
        edge_weights[(src, dst)] += 1.0
    
    # Initialize PageRank
    pr = np.ones(n) / n
    for _ in range(iterations):
        new_pr = np.ones(n) * (1 - damping) / n
        for src, neighbors in out_edges.items():
            if neighbors:
                total_weight = sum(edge_weights[(src, dst)] for dst in neighbors)
                for dst in neighbors:
                    weight = edge_weights[(src, dst)] / total_weight
                    new_pr[account_idx[dst]] += damping * pr[account_idx[src]] * weight
        pr = new_pr
    
    return {acc: float(pr[i]) for i, acc in enumerate(accounts)}


def _detect_cycles(df: pd.DataFrame, accounts: list[str], max_length: int = 4) -> dict[str, int]:
    """Count cycle participation for each account (simplified DFS-based)."""
    account_idx = {acc: i for i, acc in enumerate(accounts)}
    adj = defaultdict(set)
    for _, row in df.iterrows():
        adj[row["sender_id"]].add(row["receiver_id"])
    
    cycle_counts = defaultdict(int)
    
    def dfs(start: str, current: str, visited: set, depth: int):
        if depth > max_length:
            return
        for neighbor in adj[current]:
            if neighbor == start and depth >= 2:
                # Found a cycle
                for node in visited:
                    cycle_counts[node] += 1
            elif neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, visited, depth + 1)
                visited.remove(neighbor)
    
    for account in accounts[:min(200, len(accounts))]:  # Limit for performance
        dfs(account, account, {account}, 0)
    
    return dict(cycle_counts)


def _compute_velocity_features(df: pd.DataFrame, accounts: list[str]) -> dict[str, dict]:
    """Compute transaction velocity and burst detection features."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    velocity_features = {}
    for account in accounts:
        outgoing = df[df["sender_id"] == account].sort_values("timestamp")
        incoming = df[df["receiver_id"] == account].sort_values("timestamp")
        all_tx = pd.concat([outgoing, incoming]).sort_values("timestamp")
        
        if len(all_tx) < 2:
            velocity_features[account] = {
                "tx_velocity_per_hour": 0.0,
                "burst_score": 0.0,
                "max_hourly_count": 0,
                "dormancy_ratio": 1.0,
                "activity_concentration": 0.0,
            }
            continue
        
        # Overall velocity
        time_span_hours = (all_tx["timestamp"].max() - all_tx["timestamp"].min()).total_seconds() / 3600
        tx_velocity = len(all_tx) / max(time_span_hours, 1)
        
        # Burst detection: max transactions in any 1-hour window
        all_tx["hour_bucket"] = all_tx["timestamp"].dt.floor("h")  # lowercase 'h' for newer pandas
        hourly_counts = all_tx.groupby("hour_bucket").size()
        max_hourly = int(hourly_counts.max()) if len(hourly_counts) > 0 else 0
        
        # Burst score: ratio of max hourly to average
        avg_hourly = hourly_counts.mean() if len(hourly_counts) > 0 else 0
        burst_score = max_hourly / max(avg_hourly, 1)
        
        # Dormancy: fraction of hours with no activity
        total_hours = max(int(time_span_hours), 1)
        active_hours = len(hourly_counts)
        dormancy_ratio = 1 - (active_hours / total_hours) if total_hours > 0 else 1.0
        
        # Activity concentration: Gini-like measure
        if len(hourly_counts) > 1:
            sorted_counts = np.sort(hourly_counts.values)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            activity_concentration = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            activity_concentration = 0.0
        
        velocity_features[account] = {
            "tx_velocity_per_hour": float(tx_velocity),
            "burst_score": float(burst_score),
            "max_hourly_count": max_hourly,
            "dormancy_ratio": float(dormancy_ratio),
            "activity_concentration": float(activity_concentration),
        }
    
    return velocity_features


def _compute_structuring_features(df: pd.DataFrame, accounts: list[str], threshold: float = 10000.0) -> dict[str, dict]:
    """Detect structuring patterns (transactions just below reporting thresholds)."""
    structuring_features = {}
    
    # Common AML thresholds
    thresholds = [threshold, threshold * 0.5, threshold * 0.25]
    near_margin = 0.1  # Within 10% below threshold
    
    for account in accounts:
        outgoing = df[df["sender_id"] == account]
        
        if len(outgoing) == 0:
            structuring_features[account] = {
                "near_threshold_ratio": 0.0,
                "round_number_ratio": 0.0,
                "amount_variance_ratio": 0.0,
                "structuring_score": 0.0,
            }
            continue
        
        amounts = outgoing["transaction_amount"].values
        
        # Near-threshold detection
        near_threshold_count = 0
        for t in thresholds:
            lower = t * (1 - near_margin)
            near_threshold_count += np.sum((amounts >= lower) & (amounts < t))
        near_threshold_ratio = near_threshold_count / len(amounts)
        
        # Round number detection (multiples of 100, 500, 1000)
        round_100 = np.sum(np.mod(amounts, 100) == 0)
        round_500 = np.sum(np.mod(amounts, 500) == 0)
        round_1000 = np.sum(np.mod(amounts, 1000) == 0)
        round_number_ratio = (round_100 + round_500 * 2 + round_1000 * 3) / (len(amounts) * 6)
        
        # Amount variance (low variance = suspicious repetitive amounts)
        if len(amounts) > 1:
            cv = np.std(amounts) / (np.mean(amounts) + 1e-8)
            amount_variance_ratio = 1 / (1 + cv)  # Higher when variance is low
        else:
            amount_variance_ratio = 0.0
        
        # Combined structuring score
        structuring_score = (near_threshold_ratio * 0.5 + round_number_ratio * 0.3 + amount_variance_ratio * 0.2)
        
        structuring_features[account] = {
            "near_threshold_ratio": float(near_threshold_ratio),
            "round_number_ratio": float(round_number_ratio),
            "amount_variance_ratio": float(amount_variance_ratio),
            "structuring_score": float(structuring_score),
        }
    
    return structuring_features


def _compute_temporal_window_features(df: pd.DataFrame, accounts: list[str]) -> dict[str, dict]:
    """Compute rolling window statistics (1h, 6h, 24h windows)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    window_features = {}
    windows_minutes = {"1h": 60, "6h": 360, "24h": 1440}
    
    for account in accounts:
        all_tx = pd.concat([
            df[df["sender_id"] == account].assign(direction="out"),
            df[df["receiver_id"] == account].assign(direction="in")
        ]).sort_values("timestamp")
        
        if len(all_tx) < 2:
            features = {}
            for window_name in windows_minutes:
                features[f"tx_count_{window_name}_max"] = 0
                features[f"volume_{window_name}_max"] = 0.0
                features[f"unique_counterparties_{window_name}_max"] = 0
            window_features[account] = features
            continue
        
        features = {}
        for window_name, minutes in windows_minutes.items():
            window_td = pd.Timedelta(minutes=minutes)
            
            # Sliding window statistics
            max_count = 0
            max_volume = 0.0
            max_counterparties = 0
            
            for i, (_, row) in enumerate(all_tx.iterrows()):
                window_start = row["timestamp"] - window_td
                window_tx = all_tx[(all_tx["timestamp"] >= window_start) & (all_tx["timestamp"] <= row["timestamp"])]
                
                count = len(window_tx)
                volume = window_tx["transaction_amount"].sum()
                counterparties = len(set(window_tx["sender_id"]).union(set(window_tx["receiver_id"]))) - 1
                
                max_count = max(max_count, count)
                max_volume = max(max_volume, volume)
                max_counterparties = max(max_counterparties, counterparties)
            
            features[f"tx_count_{window_name}_max"] = max_count
            features[f"volume_{window_name}_max"] = float(max_volume)
            features[f"unique_counterparties_{window_name}_max"] = max_counterparties
        
        window_features[account] = features
    
    return window_features


def _build_node_features(df: pd.DataFrame, accounts: list[str]) -> pd.DataFrame:
    """Build comprehensive node features including advanced AML-specific features."""
    sender_group = df.groupby("sender_id")
    receiver_group = df.groupby("receiver_id")
    
    # Compute advanced features
    pagerank_scores = _compute_pagerank(df, accounts)
    cycle_counts = _detect_cycles(df, accounts)
    velocity_features = _compute_velocity_features(df, accounts)
    structuring_features = _compute_structuring_features(df, accounts)
    window_features = _compute_temporal_window_features(df, accounts)
    
    features = []
    for account in accounts:
        outgoing = sender_group.get_group(account) if account in sender_group.groups else pd.DataFrame(columns=df.columns)
        incoming = receiver_group.get_group(account) if account in receiver_group.groups else pd.DataFrame(columns=df.columns)
        
        # Basic features
        base_features = {
            "account_id": account,
            "out_degree": len(outgoing),
            "in_degree": len(incoming),
            "out_volume": float(outgoing["transaction_amount"].sum()) if len(outgoing) else 0.0,
            "in_volume": float(incoming["transaction_amount"].sum()) if len(incoming) else 0.0,
            "avg_out_amount": float(outgoing["transaction_amount"].mean()) if len(outgoing) else 0.0,
            "avg_in_amount": float(incoming["transaction_amount"].mean()) if len(incoming) else 0.0,
            "unique_receivers": outgoing["receiver_id"].nunique() if len(outgoing) else 0,
            "unique_senders": incoming["sender_id"].nunique() if len(incoming) else 0,
            "fraud_out_ratio": float(outgoing["label"].mean()) if len(outgoing) else 0.0,
            "fraud_in_ratio": float(incoming["label"].mean()) if len(incoming) else 0.0,
        }
        
        # Graph structure features
        base_features["pagerank"] = pagerank_scores.get(account, 0.0)
        base_features["cycle_participation"] = cycle_counts.get(account, 0)
        
        # In/out ratio (flow imbalance)
        total_in = base_features["in_volume"]
        total_out = base_features["out_volume"]
        base_features["flow_imbalance"] = (total_out - total_in) / (total_out + total_in + 1e-8)
        
        # Velocity features
        vel = velocity_features.get(account, {})
        base_features["tx_velocity_per_hour"] = vel.get("tx_velocity_per_hour", 0.0)
        base_features["burst_score"] = vel.get("burst_score", 0.0)
        base_features["max_hourly_count"] = vel.get("max_hourly_count", 0)
        base_features["dormancy_ratio"] = vel.get("dormancy_ratio", 1.0)
        base_features["activity_concentration"] = vel.get("activity_concentration", 0.0)
        
        # Structuring features
        struct = structuring_features.get(account, {})
        base_features["near_threshold_ratio"] = struct.get("near_threshold_ratio", 0.0)
        base_features["round_number_ratio"] = struct.get("round_number_ratio", 0.0)
        base_features["amount_variance_ratio"] = struct.get("amount_variance_ratio", 0.0)
        base_features["structuring_score"] = struct.get("structuring_score", 0.0)
        
        # Temporal window features
        win = window_features.get(account, {})
        base_features["tx_count_1h_max"] = win.get("tx_count_1h_max", 0)
        base_features["tx_count_6h_max"] = win.get("tx_count_6h_max", 0)
        base_features["tx_count_24h_max"] = win.get("tx_count_24h_max", 0)
        base_features["volume_1h_max"] = win.get("volume_1h_max", 0.0)
        base_features["volume_6h_max"] = win.get("volume_6h_max", 0.0)
        base_features["volume_24h_max"] = win.get("volume_24h_max", 0.0)
        base_features["unique_counterparties_1h_max"] = win.get("unique_counterparties_1h_max", 0)
        base_features["unique_counterparties_6h_max"] = win.get("unique_counterparties_6h_max", 0)
        base_features["unique_counterparties_24h_max"] = win.get("unique_counterparties_24h_max", 0)
        
        features.append(base_features)
    
    return pd.DataFrame(features)


def _time_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["sender_prev_time"] = df.groupby("sender_id")["timestamp"].shift(1)
    df["receiver_prev_time"] = df.groupby("receiver_id")["timestamp"].shift(1)
    df["sender_gap_minutes"] = (
        (df["timestamp"] - df["sender_prev_time"]).dt.total_seconds().fillna(0.0) / 60.0
    )
    df["receiver_gap_minutes"] = (
        (df["timestamp"] - df["receiver_prev_time"]).dt.total_seconds().fillna(0.0) / 60.0
    )
    return df


def _engineer_transaction_features(df: pd.DataFrame, node_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = _time_gap_features(df)
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)

    sender_features = node_frame.add_prefix("sender_").rename(columns={"sender_account_id": "sender_id"})
    receiver_features = node_frame.add_prefix("receiver_").rename(columns={"receiver_account_id": "receiver_id"})
    frame = frame.merge(sender_features, on="sender_id", how="left")
    frame = frame.merge(receiver_features, on="receiver_id", how="left")
    frame = frame.fillna(0.0)

    # Extended feature columns with all new advanced features
    feature_columns = [
        "transaction_amount_scaled",
        "transaction_type_encoded",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "sender_gap_minutes_scaled",
        "receiver_gap_minutes_scaled",
        # Basic node features
        "sender_out_degree",
        "sender_in_degree",
        "sender_out_volume",
        "sender_in_volume",
        "sender_avg_out_amount",
        "sender_avg_in_amount",
        "sender_unique_receivers",
        "sender_unique_senders",
        "receiver_out_degree",
        "receiver_in_degree",
        "receiver_out_volume",
        "receiver_in_volume",
        "receiver_avg_out_amount",
        "receiver_avg_in_amount",
        "receiver_unique_receivers",
        "receiver_unique_senders",
        # Graph structure features
        "sender_pagerank",
        "sender_cycle_participation",
        "sender_flow_imbalance",
        "receiver_pagerank",
        "receiver_cycle_participation",
        "receiver_flow_imbalance",
        # Velocity features
        "sender_tx_velocity_per_hour",
        "sender_burst_score",
        "sender_max_hourly_count",
        "sender_dormancy_ratio",
        "sender_activity_concentration",
        "receiver_tx_velocity_per_hour",
        "receiver_burst_score",
        "receiver_max_hourly_count",
        "receiver_dormancy_ratio",
        "receiver_activity_concentration",
        # Structuring features
        "sender_near_threshold_ratio",
        "sender_round_number_ratio",
        "sender_amount_variance_ratio",
        "sender_structuring_score",
        "receiver_near_threshold_ratio",
        "receiver_round_number_ratio",
        "receiver_amount_variance_ratio",
        "receiver_structuring_score",
        # Temporal window features
        "sender_tx_count_1h_max",
        "sender_tx_count_6h_max",
        "sender_tx_count_24h_max",
        "sender_volume_1h_max",
        "sender_volume_6h_max",
        "sender_volume_24h_max",
        "sender_unique_counterparties_1h_max",
        "sender_unique_counterparties_6h_max",
        "sender_unique_counterparties_24h_max",
        "receiver_tx_count_1h_max",
        "receiver_tx_count_6h_max",
        "receiver_tx_count_24h_max",
        "receiver_volume_1h_max",
        "receiver_volume_6h_max",
        "receiver_volume_24h_max",
        "receiver_unique_counterparties_1h_max",
        "receiver_unique_counterparties_6h_max",
        "receiver_unique_counterparties_24h_max",
    ]
    return frame, feature_columns


def _build_temporal_sequences(train_df: pd.DataFrame, test_df: pd.DataFrame, account_to_index: dict[str, int]) -> dict:
    """Build temporal sequences for TGN-style training with chronological batching."""
    
    def extract_sequences(df: pd.DataFrame) -> dict:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Extract normalized timestamps (seconds since first transaction)
        min_time = df["timestamp"].min()
        df["time_delta"] = (df["timestamp"] - min_time).dt.total_seconds()
        
        # Build edge sequences in chronological order
        sources = df["sender_id"].map(account_to_index).values
        destinations = df["receiver_id"].map(account_to_index).values
        timestamps = df["time_delta"].values
        labels = df["label"].values
        
        return {
            "sources": sources,
            "destinations": destinations,
            "timestamps": timestamps,
            "labels": labels,
            "num_edges": len(df),
        }
    
    return {
        "train": extract_sequences(train_df),
        "test": extract_sequences(test_df),
        "num_nodes": len(account_to_index),
    }


def _build_graph_batch(
    df: pd.DataFrame, node_features: np.ndarray, account_to_index: dict[str, int], edge_feature_names: list[str]
) -> GraphBatch:
    edge_index_array = np.vstack(
        [
            df["sender_id"].map(account_to_index).to_numpy(dtype=np.int64),
            df["receiver_id"].map(account_to_index).to_numpy(dtype=np.int64),
        ]
    )
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)
    edge_attr = torch.tensor(df[edge_feature_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    labels = torch.tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    edge_ids = torch.arange(len(df), dtype=torch.long)
    return GraphBatch(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
        edge_ids=edge_ids,
    )


def prepare_dataset(csv_path: str | Path, artifact_dir: str | Path) -> PreparedData:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sender_id", "receiver_id", "transaction_amount", "timestamp", "label"]).copy()
    df["transaction_type"] = df["transaction_type"].fillna("unknown")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    accounts = sorted(set(df["sender_id"]).union(df["receiver_id"]))
    account_to_index = {account: idx for idx, account in enumerate(accounts)}

    node_frame = _build_node_features(df, accounts)
    node_feature_names = [column for column in node_frame.columns if column != "account_id"]
    node_scaler = StandardScaler()
    node_matrix = node_scaler.fit_transform(node_frame[node_feature_names])

    type_encoder = LabelEncoder()
    df["transaction_type_encoded"] = type_encoder.fit_transform(df["transaction_type"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    amount_scaler = StandardScaler()
    gap_scaler = StandardScaler()
    train_df["transaction_amount_scaled"] = amount_scaler.fit_transform(train_df[["transaction_amount"]])
    test_df["transaction_amount_scaled"] = amount_scaler.transform(test_df[["transaction_amount"]])

    train_df = _time_gap_features(train_df)
    test_df = _time_gap_features(test_df)
    train_df[["sender_gap_minutes_scaled", "receiver_gap_minutes_scaled"]] = gap_scaler.fit_transform(
        train_df[["sender_gap_minutes", "receiver_gap_minutes"]]
    )
    test_df[["sender_gap_minutes_scaled", "receiver_gap_minutes_scaled"]] = gap_scaler.transform(
        test_df[["sender_gap_minutes", "receiver_gap_minutes"]]
    )

    train_df, transaction_feature_names = _engineer_transaction_features(train_df, node_frame)
    test_df, _ = _engineer_transaction_features(test_df, node_frame)

    x_train = train_df[transaction_feature_names].to_numpy(dtype=np.float32)
    x_test = test_df[transaction_feature_names].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    edge_feature_names = [
        "transaction_amount_scaled",
        "transaction_type_encoded",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "sender_gap_minutes_scaled",
        "receiver_gap_minutes_scaled",
    ]

    train_graph = _build_graph_batch(train_df, node_matrix, account_to_index, edge_feature_names)
    test_graph = _build_graph_batch(test_df, node_matrix, account_to_index, edge_feature_names)

    train_df.to_csv(artifact_root / "train_transactions.csv", index=False)
    test_df.to_csv(artifact_root / "test_transactions.csv", index=False)
    node_frame.to_csv(artifact_root / "node_features.csv", index=False)

    # Build temporal sequences for TGN training
    temporal_sequences = _build_temporal_sequences(train_df, test_df, account_to_index)

    metadata = {
        "transaction_feature_names": transaction_feature_names,
        "edge_feature_names": edge_feature_names,
        "node_feature_names": node_feature_names,
        "accounts": accounts,
        "label_mapping": {"0": "non_fraud", "1": "fraud"},
    }
    (artifact_root / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    with open(artifact_root / "preprocessors.pkl", "wb") as handle:
        pickle.dump(
            {
                "amount_scaler": amount_scaler,
                "gap_scaler": gap_scaler,
                "node_scaler": node_scaler,
                "type_encoder": type_encoder,
                "account_to_index": account_to_index,
                "node_frame": node_frame,
                "node_matrix": node_matrix,
                "transaction_feature_names": transaction_feature_names,
                "edge_feature_names": edge_feature_names,
            },
            handle,
        )

    return PreparedData(
        dataframe=df,
        train_df=train_df,
        test_df=test_df,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_graph=train_graph,
        test_graph=test_graph,
        node_frame=node_frame,
        account_to_index=account_to_index,
        transaction_feature_names=transaction_feature_names,
        node_feature_names=node_feature_names,
        temporal_sequences=temporal_sequences,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "synthetic_aml_transactions.csv"
    artifact_dir = project_root / "artifacts" / "processed"
    
    print(f"\n[1/4] Loading dataset from {data_path.name}...")
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df):,} transactions")
    print(f"  ✓ Fraud rate: {df['label'].mean()*100:.1f}%")
    
    print("\n[2/4] Building graph and computing features...")
    print("  • PageRank (20 iterations)")
    print("  • Cycle detection (max depth 4)")
    print("  • Velocity features (burst, dormancy)")
    print("  • Structuring patterns (threshold, round numbers)")
    print("  • Temporal windows (1h, 6h, 24h)")
    
    prepared = prepare_dataset(data_path, artifact_dir)
    
    print(f"\n[3/4] Feature engineering complete:")
    print(f"  ✓ Node features: {len(prepared.node_feature_names)}")
    print(f"  ✓ Transaction features: {len(prepared.transaction_feature_names)}")
    print(f"  ✓ Unique accounts: {len(prepared.account_to_index)}")
    
    print(f"\n[4/4] Train/test split:")
    print(f"  ✓ Training samples: {len(prepared.train_df):,}")
    print(f"  ✓ Test samples: {len(prepared.test_df):,}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE ✓")
    print("=" * 60)
    print(f"\nArtifacts saved to: {artifact_dir}/")
    print("  • node_features.csv")
    print("  • feature_metadata.json")
    print("  • preprocessors.pkl")
    print("  • train_graph_batch.pkl")
    print("  • test_graph_batch.pkl")
    print("  • temporal_sequences.pkl")
    print("\nNext step: python training/train.py")


