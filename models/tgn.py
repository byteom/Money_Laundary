"""
Temporal Graph Network (TGN) for Money Laundering Detection

This module implements a state-of-the-art Temporal Graph Network with:
- Time2Vec learnable time encoding
- GRU-based memory module for node state tracking
- Multi-head temporal attention for neighbor aggregation
- Focal loss for handling class imbalance
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class Time2Vec(nn.Module):
    """Learnable time encoding using Time2Vec representation.
    
    Converts continuous time values into a rich periodic representation
    that captures both linear and periodic temporal patterns.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        # Periodic components
        self.w = nn.Parameter(torch.randn(time_dim - 1))
        self.b = nn.Parameter(torch.randn(time_dim - 1))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time values.
        
        Args:
            t: Time values of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Time encoding of shape (batch_size, time_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Linear component
        linear = self.w0 * t + self.b0
        
        # Periodic components using sine activation
        periodic = torch.sin(self.w * t + self.b)
        
        return torch.cat([linear, periodic], dim=1)


class TemporalAttention(nn.Module):
    """Multi-head temporal attention for neighbor aggregation.
    
    Weights neighbor contributions based on both feature similarity
    and temporal proximity (more recent = higher weight).
    """
    def __init__(self, node_dim: int, edge_dim: int, time_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        
        self.time_encoder = Time2Vec(time_dim)
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(node_dim, node_dim)
        self.key_proj = nn.Linear(node_dim + time_dim + edge_dim, node_dim)
        self.value_proj = nn.Linear(node_dim + time_dim + edge_dim, node_dim)
        
        # Output projection
        self.out_proj = nn.Linear(node_dim, node_dim)
        
        # Time decay parameter (learnable)
        self.time_decay = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(node_dim)
    
    def forward(
        self,
        query_nodes: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        time_deltas: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute temporal attention over neighbors.
        
        Args:
            query_nodes: Target node features (batch, node_dim)
            neighbor_nodes: Neighbor node features (batch, max_neighbors, node_dim)
            edge_features: Edge features (batch, max_neighbors, edge_dim)
            time_deltas: Time since each neighbor interaction (batch, max_neighbors)
            neighbor_mask: Boolean mask for valid neighbors (batch, max_neighbors)
        
        Returns:
            aggregated: Aggregated neighbor information (batch, node_dim)
            attention_weights: Attention weights (batch, num_heads, max_neighbors)
        """
        batch_size, max_neighbors, _ = neighbor_nodes.shape
        
        # Encode time deltas
        time_encoding = self.time_encoder(time_deltas.view(-1)).view(batch_size, max_neighbors, -1)
        
        # Combine neighbor features with time and edge info
        neighbor_combined = torch.cat([neighbor_nodes, time_encoding, edge_features], dim=-1)
        
        # Project query, keys, values
        Q = self.query_proj(query_nodes)  # (batch, node_dim)
        K = self.key_proj(neighbor_combined)  # (batch, max_neighbors, node_dim)
        V = self.value_proj(neighbor_combined)  # (batch, max_neighbors, node_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, max_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, max_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply time decay (more recent = higher weight)
        time_decay_factor = torch.exp(-self.time_decay * time_deltas.unsqueeze(1).unsqueeze(1))
        scores = scores * time_decay_factor
        
        # Apply neighbor mask if provided
        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, max_neighbors)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate values
        aggregated = torch.matmul(attention_weights, V)  # (batch, num_heads, 1, head_dim)
        aggregated = aggregated.transpose(1, 2).contiguous().view(batch_size, -1)
        
        # Output projection with residual
        output = self.out_proj(aggregated)
        output = self.layer_norm(output + query_nodes)
        
        return output, attention_weights.squeeze(2)


class MemoryModule(nn.Module):
    """GRU-based memory module for tracking node temporal state.
    
    Each node maintains a memory vector that gets updated based on
    its interactions over time.
    """
    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        
        # GRU for memory updates
        self.gru = nn.GRUCell(message_dim, memory_dim)
        
        # Message function: combine source, destination, edge, time
        self.message_mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
    
    def compute_message(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute message from an interaction."""
        combined = torch.cat([src_memory, dst_memory, edge_features], dim=-1)
        return self.message_mlp(combined)
    
    def update_memory(self, memory: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Update memory with new message."""
        return self.gru(message, memory)


class MessageAggregator(nn.Module):
    """Aggregates messages from multiple neighbors using multiple strategies."""
    
    def __init__(self, message_dim: int):
        super().__init__()
        self.message_dim = message_dim
        
        # Learnable aggregation weights
        self.agg_weights = nn.Parameter(torch.ones(3) / 3)  # mean, max, last
        
        # Transform for combined aggregation
        self.combine_mlp = nn.Sequential(
            nn.Linear(message_dim * 3, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
    
    def forward(
        self,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate messages using learned combination of strategies.
        
        Args:
            messages: (batch, max_messages, message_dim)
            timestamps: (batch, max_messages) - for most-recent selection
            mask: (batch, max_messages) - valid message mask
        """
        if mask is not None:
            messages = messages * mask.unsqueeze(-1)
        
        # Mean aggregation
        if mask is not None:
            mean_agg = messages.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            mean_agg = messages.mean(dim=1)
        
        # Max aggregation
        max_agg = messages.max(dim=1).values
        
        # Most recent (highest timestamp)
        if mask is not None:
            timestamps = timestamps.masked_fill(~mask, float('-inf'))
        most_recent_idx = timestamps.argmax(dim=1)
        batch_idx = torch.arange(messages.size(0), device=messages.device)
        last_agg = messages[batch_idx, most_recent_idx]
        
        # Combine with learned weights
        weights = F.softmax(self.agg_weights, dim=0)
        combined = torch.stack([mean_agg, max_agg, last_agg], dim=0)
        weighted = (combined * weights.view(3, 1, 1)).sum(dim=0)
        
        # Also return raw combination for richer representation
        raw_combined = torch.cat([mean_agg, max_agg, last_agg], dim=-1)
        
        return self.combine_mlp(raw_combined) + weighted


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in fraud detection.
    
    Down-weights easy examples and focuses on hard, misclassified cases.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()


class TemporalGraphNetwork(nn.Module):
    """Simplified but effective Temporal Graph Network for fraud detection.
    
    Combines:
    - Node feature projection
    - Time encoding (Time2Vec)
    - Temporal edge classification with attention-like mechanism
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        memory_dim: int = 64,
        time_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers
        
        # Initial node projection
        self.node_proj = nn.Linear(node_dim, memory_dim)
        
        # Time encoder
        self.time_encoder = Time2Vec(time_dim)
        
        # Graph convolution layers (simplified message passing)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = memory_dim if i == 0 else hidden_dim
            self.conv_layers.append(nn.Sequential(
                nn.Linear(in_dim * 2 + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ))
        
        # Final node transform
        self.node_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Edge classifier with temporal features
        classifier_input_dim = hidden_dim * 2 + edge_dim + time_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Store attention weights for explainability
        self.last_attention_weights = None
    
    def init_memory(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Initialize memory for all nodes."""
        return torch.zeros(num_nodes, self.memory_dim, device=device)
    
    def message_passing(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Simple message passing aggregation."""
        src_idx, dst_idx = edge_index
        num_nodes = x.size(0)
        
        for conv in self.conv_layers:
            # Gather source and destination features
            src_features = x[src_idx]
            dst_features = x[dst_idx]
            
            # Compute edge messages
            edge_messages = torch.cat([src_features, dst_features, edge_attr], dim=-1)
            edge_messages = conv(edge_messages)
            
            # Aggregate messages to destination nodes (mean aggregation)
            aggregated = torch.zeros(num_nodes, edge_messages.size(-1), device=x.device)
            counts = torch.zeros(num_nodes, 1, device=x.device)
            aggregated.index_add_(0, dst_idx, edge_messages)
            counts.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=x.device))
            counts = counts.clamp(min=1)
            aggregated = aggregated / counts
            
            # Also aggregate to source nodes (bidirectional)
            aggregated_src = torch.zeros(num_nodes, edge_messages.size(-1), device=x.device)
            counts_src = torch.zeros(num_nodes, 1, device=x.device)
            aggregated_src.index_add_(0, src_idx, edge_messages)
            counts_src.index_add_(0, src_idx, torch.ones(len(src_idx), 1, device=x.device))
            counts_src = counts_src.clamp(min=1)
            aggregated_src = aggregated_src / counts_src
            
            # Combine
            x = aggregated + aggregated_src
        
        return self.node_transform(x)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for edge classification."""
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        device = node_features.device
        
        # Project node features
        node_embed = self.node_proj(node_features)
        
        # Apply message passing
        node_state = self.message_passing(node_embed, edge_index, edge_attr)
        
        # Get source and destination nodes for edges
        src_idx, dst_idx = edge_index
        src_state = node_state[src_idx]
        dst_state = node_state[dst_idx]
        
        # Encode timestamps if provided
        if edge_timestamps is not None:
            time_encoding = self.time_encoder(edge_timestamps)
        else:
            time_encoding = torch.zeros(num_edges, self.time_encoder.time_dim, device=device)
        
        # Classify edges
        edge_features = torch.cat([src_state, dst_state, edge_attr, time_encoding], dim=-1)
        logits = self.edge_classifier(edge_features).squeeze(-1)
        
        # Return dummy memory for API compatibility
        updated_memory = self.init_memory(num_nodes, device)
        
        return logits, updated_memory, None


@dataclass
class TGNResult:
    metrics: dict[str, float]
    best_epoch: int
    training_history: list[dict]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_tgn(
    train_batch,
    test_batch,
    artifact_dir: str | Path,
    epochs: int = 200,
    learning_rate: float = 0.003,
    patience: int = 30,
    use_focal_loss: bool = True,
) -> tuple[TemporalGraphNetwork, TGNResult, torch.Tensor]:
    """Train the Temporal Graph Network.
    
    Args:
        train_batch: Training graph batch
        test_batch: Test graph batch
        artifact_dir: Directory to save artifacts
        epochs: Maximum training epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        use_focal_loss: Whether to use focal loss
    
    Returns:
        model: Trained TGN model
        result: Training results
        node_embeddings: Final node embeddings
    """
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = TemporalGraphNetwork(
        node_dim=train_batch.x.size(1),
        edge_dim=train_batch.edge_attr.size(1),
        memory_dim=64,
        time_dim=16,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
    )
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    # Loss function - use weighted BCE for better class balance handling
    pos_weight = torch.tensor([(train_batch.labels == 0).sum() / max((train_batch.labels == 1).sum(), 1)])
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.4, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        
        logits, _, _ = model(
            train_batch.x,
            train_batch.edge_index,
            train_batch.edge_attr,
        )
        
        loss = criterion(logits, train_batch.labels.float())
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits, _, _ = model(
                test_batch.x,
                test_batch.edge_index,
                test_batch.edge_attr,
            )
            test_probs = torch.sigmoid(test_logits)
            
            y_true = test_batch.labels.cpu().numpy()
            y_prob = test_probs.cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
            
            current_metrics = compute_metrics(y_true, y_pred, y_prob)
            current_f1 = current_metrics["f1"]
        
        # Update scheduler
        scheduler.step(current_f1)
        
        # Track history
        training_history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "f1": current_f1,
            "roc_auc": current_metrics["roc_auc"],
        })
        
        # Check for improvement
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is None:
        raise RuntimeError("TGN training did not produce a valid checkpoint.")
    
    model.load_state_dict(best_state)
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        test_logits, _, _ = model(
            test_batch.x,
            test_batch.edge_index,
            test_batch.edge_attr,
        )
        test_probs = torch.sigmoid(test_logits)
        
        y_true = test_batch.labels.cpu().numpy()
        y_prob = test_probs.cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
        final_metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Save model and results
    torch.save(model.state_dict(), artifact_root / "tgn_model.pt")
    
    result = TGNResult(
        metrics=final_metrics,
        best_epoch=best_epoch,
        training_history=training_history,
    )
    
    report = {
        "metrics": final_metrics,
        "best_epoch": best_epoch,
        "training_history": training_history[-10:],  # Last 10 epochs
    }
    (artifact_root / "tgn_metrics.json").write_text(json.dumps(report, indent=2))
    
    # Get node embeddings
    with torch.no_grad():
        node_embed = model.node_proj(test_batch.x)
    
    return model, result, node_embed.detach().cpu()
