from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_layers: int = 0, dropout: float = 0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = int(self.transformer.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._freeze_bottom_layers(freeze_layers)

    def _freeze_bottom_layers(self, count: int) -> None:
        if count <= 0:
            return
        embeddings = getattr(self.transformer, "embeddings", None)
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = False
        stack = getattr(self.transformer, "transformer", None)
        layers = getattr(stack, "layer", None)
        if layers is None:
            encoder = getattr(self.transformer, "encoder", None)
            layers = getattr(encoder, "layer", [])
        for layer in list(layers)[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.transformer(**tokens)
        sequence = output.last_hidden_state
        cls = self.dropout(sequence[:, 0])
        return cls, sequence


class BertTagClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_layers: int = 0, dropout: float = 0.3):
        super().__init__()
        self.text_encoder = TextEncoder(model_name, freeze_layers, dropout)
        self.classifier = nn.Linear(self.text_encoder.output_dim, num_labels)

    def forward(self, tokens: dict[str, torch.Tensor], **_: dict) -> dict[str, torch.Tensor]:
        cls, sequence = self.text_encoder(tokens)
        return {"logits": self.classifier(cls), "embedding": cls, "token_embeddings": sequence}

