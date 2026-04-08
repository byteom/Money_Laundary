from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    labels: torch.Tensor
    edge_ids: torch.Tensor


def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, reverse_edges], dim=1)


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    messages = x[src]
    out = torch.zeros_like(x)
    counts = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
    out.index_add_(0, dst, messages)
    counts.index_add_(0, dst, torch.ones((messages.size(0), 1), dtype=x.dtype, device=x.device))
    counts = counts.clamp_min(1.0)
    return out / counts
