from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from graph.temporal_graph import GraphBatch, make_undirected, mean_aggregate


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        undirected_edges = make_undirected(edge_index)
        neighbor_mean = mean_aggregate(x, undirected_edges)
        combined = torch.cat([x, neighbor_mean], dim=1)
        return self.activation(self.linear(combined))


class TemporalGraphSAGE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = GraphSAGELayer(node_dim, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        return h

    def score_edges_from_embeddings(
        self, node_embeddings: torch.Tensor, edge_pairs: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src = node_embeddings[edge_pairs[0]]
        dst = node_embeddings[edge_pairs[1]]
        features = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(features).squeeze(1)

    def forward(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode_nodes(batch.x, batch.edge_index)
        logits = self.score_edges_from_embeddings(embeddings, batch.edge_index, batch.edge_attr)
        return logits, embeddings


@dataclass
class GNNResult:
    metrics: dict[str, float]
    best_epoch: int


def _metrics(y_true: torch.Tensor, y_prob: torch.Tensor) -> dict[str, float]:
    y_np = y_true.detach().cpu().numpy()
    prob_np = y_prob.detach().cpu().numpy()
    pred_np = (prob_np >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_np, pred_np)),
        "precision": float(precision_score(y_np, pred_np, zero_division=0)),
        "recall": float(recall_score(y_np, pred_np, zero_division=0)),
        "f1": float(f1_score(y_np, pred_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_np, prob_np)),
    }


def train_gnn(
    train_batch: GraphBatch,
    test_batch: GraphBatch,
    artifact_dir: str | Path,
    epochs: int = 120,
    learning_rate: float = 0.003,
) -> tuple[TemporalGraphSAGE, GNNResult, torch.Tensor]:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    model = TemporalGraphSAGE(
        node_dim=train_batch.x.size(1),
        edge_dim=train_batch.edge_attr.size(1),
        hidden_dim=64,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(train_batch)
        loss = criterion(logits, train_batch.labels.float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits, _ = model(test_batch)
            probabilities = torch.sigmoid(test_logits)
            current_f1 = f1_score(
                test_batch.labels.detach().cpu().numpy(),
                (probabilities.detach().cpu().numpy() >= 0.5).astype(int),
                zero_division=0,
            )

        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("GNN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(test_batch)
        test_prob = torch.sigmoid(test_logits)
        metrics = _metrics(test_batch.labels, test_prob)
        _, test_embeddings = model(test_batch)

    torch.save(model.state_dict(), artifact_root / "temporal_graphsage.pt")
    report = {"metrics": metrics, "best_epoch": best_epoch}
    (artifact_root / "gnn_metrics.json").write_text(json.dumps(report, indent=2))
    return model, GNNResult(metrics=metrics, best_epoch=best_epoch), test_embeddings.detach().cpu()
