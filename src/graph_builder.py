from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def build_edge_index(
    node_features: torch.Tensor,
    similarity_threshold: float = 0.80,
    similarity_top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = node_features.size(0)
    edge_weights: dict[tuple[int, int], float] = {}

    for source in range(num_nodes - 1):
        edge_weights[(source, source + 1)] = 1.0
        edge_weights[(source + 1, source)] = 1.0

    normalized = F.normalize(node_features, p=2, dim=1)
    similarity = normalized @ normalized.T
    for source in range(num_nodes):
        candidates: list[tuple[float, int]] = []
        for target in range(num_nodes):
            if source == target or abs(source - target) == 1:
                continue
            score = float(similarity[source, target])
            if score >= similarity_threshold:
                candidates.append((score, target))
        for score, target in sorted(candidates, reverse=True)[:similarity_top_k]:
            edge_weights[(source, target)] = score
            edge_weights[(target, source)] = score

    if not edge_weights:
        edge_weights[(0, 0)] = 1.0
    edges = sorted(edge_weights)
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    edge_attr = torch.tensor([edge_weights[edge] for edge in edges], dtype=torch.float32)
    return edge_index, edge_attr


def make_graph(
    node_features: torch.Tensor,
    labels: torch.Tensor,
    track_id: int,
    text: str,
    similarity_threshold: float,
    similarity_top_k: int,
) -> Data:
    edge_index, edge_attr = build_edge_index(
        node_features, similarity_threshold=similarity_threshold, similarity_top_k=similarity_top_k
    )
    return Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels.float().view(1, -1),
        track_id=torch.tensor([int(track_id)], dtype=torch.long),
        text=text,
    )


def graph_to_json(graph: Data, label_names: list[str]) -> dict[str, Any]:
    labels = graph.y.view(-1).detach().cpu().numpy()
    return {
        "track_id": int(graph.track_id.view(-1)[0]),
        "text": str(graph.text),
        "num_nodes": int(graph.num_nodes),
        "node_feature_dimension": int(graph.num_node_features),
        "edges": graph.edge_index.T.detach().cpu().tolist(),
        "edge_weights": graph.edge_attr.detach().cpu().tolist(),
        "positive_labels": [name for name, value in zip(label_names, labels) if value > 0.5],
        "node_features_preview": graph.x[:, :8].detach().cpu().tolist(),
    }

