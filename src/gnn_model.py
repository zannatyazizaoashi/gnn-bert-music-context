from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import BatchNorm, SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        if layers < 2:
            raise ValueError("GraphEncoder requires at least two layers")
        dimensions = [input_dim] + [hidden_dim] * layers
        self.convolutions = nn.ModuleList(
            [SAGEConv(dimensions[i], dimensions[i + 1]) for i in range(layers)]
        )
        self.normalizations = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        x = graph.x
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = convolution(x, graph.edge_index)
            x = normalization(x)
            x = self.dropout(self.activation(x))
        pooled = global_mean_pool(x, graph.batch)
        return pooled, x


class GNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 192, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.graph_encoder = GraphEncoder(input_dim, hidden_dim, layers, dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, graph, **_: dict) -> dict[str, torch.Tensor]:
        graph_embedding, node_embeddings = self.graph_encoder(graph)
        return {
            "logits": self.classifier(graph_embedding),
            "embedding": graph_embedding,
            "node_embeddings": node_embeddings,
        }

