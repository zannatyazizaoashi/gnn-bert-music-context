from __future__ import annotations

import torch
from torch import nn

from .bert_encoder import TextEncoder
from .gnn_model import GraphEncoder


class EarlyConcatFusion(nn.Module):
    def __init__(
        self, graph_input_dim: int, num_labels: int, model_name: str,
        gnn_hidden_dim: int = 192, gnn_layers: int = 3,
        fusion_dim: int = 192, freeze_layers: int = 4, dropout: float = 0.3,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(graph_input_dim, gnn_hidden_dim, gnn_layers, dropout)
        self.text_encoder = TextEncoder(model_name, freeze_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim + self.text_encoder.output_dim, fusion_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(fusion_dim)
        )
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, graph, tokens: dict[str, torch.Tensor], **_: dict) -> dict[str, torch.Tensor]:
        graph_embedding, node_embeddings = self.graph_encoder(graph)
        text_embedding, token_embeddings = self.text_encoder(tokens)
        embedding = self.fusion(torch.cat([graph_embedding, text_embedding], dim=-1))
        return {
            "logits": self.classifier(embedding), "embedding": embedding,
            "node_embeddings": node_embeddings, "token_embeddings": token_embeddings,
        }


class CrossAttentionFusion(nn.Module):
    def __init__(
        self, graph_input_dim: int, num_labels: int, model_name: str,
        gnn_hidden_dim: int = 192, gnn_layers: int = 3, fusion_dim: int = 192,
        attention_heads: int = 4, freeze_layers: int = 4, dropout: float = 0.3,
    ):
        super().__init__()
        if fusion_dim % attention_heads:
            raise ValueError("fusion_dim must be divisible by attention_heads")
        self.graph_encoder = GraphEncoder(graph_input_dim, gnn_hidden_dim, gnn_layers, dropout)
        self.text_encoder = TextEncoder(model_name, freeze_layers, dropout)
        self.graph_projection = nn.Linear(gnn_hidden_dim, fusion_dim)
        self.text_projection = nn.Linear(self.text_encoder.output_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim + fusion_dim, fusion_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(fusion_dim)
        )
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, graph, tokens: dict[str, torch.Tensor], **_: dict) -> dict[str, torch.Tensor]:
        graph_embedding, node_embeddings = self.graph_encoder(graph)
        _, token_embeddings = self.text_encoder(tokens)
        query = self.graph_projection(graph_embedding).unsqueeze(1)
        keys = self.text_projection(token_embeddings)
        padding_mask = tokens["attention_mask"].eq(0)
        attended, attention = self.cross_attention(
            query=query, key=keys, value=keys,
            key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False
        )
        embedding = self.fusion(torch.cat([graph_embedding, attended.squeeze(1)], dim=-1))
        return {
            "logits": self.classifier(embedding), "embedding": embedding,
            "attention": attention.squeeze(2), "node_embeddings": node_embeddings,
            "token_embeddings": token_embeddings,
        }

