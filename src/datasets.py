from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class ManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str, mode: str):
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.mode = mode
        if mode not in {"text", "graph", "mel", "multimodal"}:
            raise ValueError(f"Unsupported dataset mode {mode}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        item: dict[str, Any] = {
            "track_id": int(row["track_id"]),
            "text": str(row["text"]),
            "labels": torch.tensor(json.loads(row["labels"]), dtype=torch.float32),
        }
        if self.mode in {"graph", "multimodal"}:
            item["graph"] = torch.load(row["graph_path"], map_location="cpu", weights_only=False)
        if self.mode == "mel":
            item["mel"] = torch.load(row["mel_path"], map_location="cpu", weights_only=False)["mel"].float()
        return item


class BatchCollator:
    def __init__(self, mode: str, tokenizer=None, max_length: int = 96):
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "track_id": torch.tensor([item["track_id"] for item in items], dtype=torch.long),
            "labels": torch.stack([item["labels"] for item in items]),
            "text": [item["text"] for item in items],
        }
        if self.mode in {"graph", "multimodal"}:
            batch["graph"] = Batch.from_data_list([item["graph"] for item in items])
        if self.mode == "mel":
            batch["mel"] = torch.stack([item["mel"] for item in items])
        if self.mode in {"text", "multimodal"}:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text batches")
            batch["tokens"] = self.tokenizer(
                batch["text"], padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
        return batch


def infer_dimensions(manifest: pd.DataFrame) -> tuple[int, int]:
    if manifest.empty:
        raise ValueError("Manifest is empty")
    graph = torch.load(manifest.iloc[0]["graph_path"], map_location="cpu", weights_only=False)
    labels = json.loads(manifest.iloc[0]["labels"])
    return int(graph.num_node_features), len(labels)
