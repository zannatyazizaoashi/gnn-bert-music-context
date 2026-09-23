from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .bert_encoder import TextEncoder
from .gnn_model import GraphEncoder


class ContrastiveGraphTextModel(nn.Module):
    def __init__(
        self, graph_input_dim: int, model_name: str, gnn_hidden_dim: int = 192,
        gnn_layers: int = 3, projection_dim: int = 128,
        freeze_layers: int = 4, dropout: float = 0.3, temperature: float = 0.07,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(graph_input_dim, gnn_hidden_dim, gnn_layers, dropout)
        self.text_encoder = TextEncoder(model_name, freeze_layers, dropout)
        self.graph_projection = nn.Linear(gnn_hidden_dim, projection_dim)
        self.text_projection = nn.Linear(self.text_encoder.output_dim, projection_dim)
        self.temperature = temperature

    def forward(self, graph, tokens: dict[str, torch.Tensor], **_: dict) -> dict[str, torch.Tensor]:
        graph_embedding, _ = self.graph_encoder(graph)
        text_embedding, _ = self.text_encoder(tokens)
        graph_embedding = F.normalize(self.graph_projection(graph_embedding), dim=-1)
        text_embedding = F.normalize(self.text_projection(text_embedding), dim=-1)
        return {"graph_embedding": graph_embedding, "text_embedding": text_embedding}

    def loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = outputs["graph_embedding"] @ outputs["text_embedding"].T / self.temperature
        targets = torch.arange(similarity.size(0), device=similarity.device)
        return 0.5 * (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets))


def retrieval_at_k(graph_embeddings: torch.Tensor, text_embeddings: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    similarity = F.normalize(graph_embeddings, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
    metrics: dict[str, float] = {}
    for direction, scores in (("text_to_graph", similarity.T), ("graph_to_text", similarity)):
        ranking = scores.argsort(dim=1, descending=True)
        targets = torch.arange(scores.size(0), device=scores.device).view(-1, 1)
        for k in ks:
            effective_k = min(k, scores.size(1))
            metrics[f"{direction}_R@{k}"] = float((ranking[:, :effective_k] == targets).any(dim=1).float().mean())
    return metrics

