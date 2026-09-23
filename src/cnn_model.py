from __future__ import annotations

import torch
from torch import nn


class MelCNNClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(192, num_labels)

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, mel: torch.Tensor, **_: dict) -> dict[str, torch.Tensor]:
        embedding = self.features(mel).flatten(1)
        embedding = self.dropout(embedding)
        return {"logits": self.classifier(embedding), "embedding": embedding}

