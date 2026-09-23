from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


def tune_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(targets, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_label_ap = average_precision_score(targets, probabilities, average=None)
        macro_ap = average_precision_score(targets, probabilities, average="macro")
        micro_ap = average_precision_score(targets, probabilities, average="micro")
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_auc_pr": float(np.nanmean(per_label_ap)),
        "micro_auc_pr": float(micro_ap),
        "per_label_auc_pr": np.nan_to_num(per_label_ap, nan=0.0).tolist(),
        "exact_match": float(np.all(targets == predictions, axis=1).mean()),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("labels", "mel", "graph", "track_id"):
        if key in moved:
            moved[key] = moved[key].to(device)
    if "tokens" in moved:
        moved["tokens"] = {key: value.to(device) for key, value in moved["tokens"].items()}
    return moved


def model_forward(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    kwargs = {key: batch[key] for key in ("mel", "graph", "tokens") if key in batch}
    return model(**kwargs)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader, device: torch.device, include_details: bool = False
) -> dict[str, Any]:
    model.eval()
    logits, labels, embeddings, track_ids = [], [], [], []
    texts: list[str] = []
    attention, token_ids = [], []
    for batch in loader:
        original_text = batch.get("text", [])
        batch = _to_device(batch, device)
        output = model_forward(model, batch)
        logits.append(output["logits"].detach().cpu())
        labels.append(batch["labels"].detach().cpu())
        track_ids.append(batch["track_id"].detach().cpu())
        if "embedding" in output:
            embeddings.append(output["embedding"].detach().cpu())
        if include_details and "attention" in output:
            attention.append(output["attention"].mean(dim=1).detach().cpu())
            token_ids.append(batch["tokens"]["input_ids"].detach().cpu())
            texts.extend(original_text)
    result: dict[str, Any] = {
        "logits": torch.cat(logits).numpy(),
        "probabilities": torch.sigmoid(torch.cat(logits)).numpy(),
        "targets": torch.cat(labels).numpy(),
        "track_ids": torch.cat(track_ids).numpy(),
    }
    if embeddings:
        result["embeddings"] = torch.cat(embeddings).numpy()
    if attention:
        result["attention"] = torch.cat(attention).numpy()
        result["token_ids"] = torch.cat(token_ids).numpy()
        result["texts"] = texts
    return result


def evaluate_with_validation_threshold(
    model: torch.nn.Module, validation_loader, test_loader, device: torch.device,
    tune: bool = True, default_threshold: float = 0.5, include_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = collect_predictions(model, validation_loader, device)
    threshold = tune_threshold(validation["targets"], validation["probabilities"]) if tune else default_threshold
    test = collect_predictions(model, test_loader, device, include_details=include_details)
    metrics = classification_metrics(test["targets"], test["probabilities"], threshold)
    metrics["validation_threshold"] = threshold
    metrics["test_examples"] = int(len(test["targets"]))
    return metrics, test

