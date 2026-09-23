from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .bert_encoder import BertTagClassifier
from .cnn_model import MelCNNClassifier
from .datasets import BatchCollator, ManifestDataset, infer_dimensions
from .evaluate import _to_device, classification_metrics, collect_predictions, model_forward, tune_threshold
from .fusion_model import CrossAttentionFusion, EarlyConcatFusion
from .gnn_model import GNNClassifier
from .utils import count_parameters, ensure_project_dirs, load_config, save_json, seed_everything


MODEL_MODES = {
    "cnn": "mel", "bert": "text", "gnn": "graph",
    "early_concat": "multimodal", "cross_attention": "multimodal",
}


def build_model(model_name: str, config: dict[str, Any], input_dim: int, num_labels: int) -> nn.Module:
    model_config = config["model"]
    common_gnn = dict(
        input_dim=input_dim, num_labels=num_labels,
        hidden_dim=model_config["gnn_hidden_dim"], layers=model_config["gnn_layers"],
        dropout=model_config["dropout"],
    )
    if model_name == "cnn":
        return MelCNNClassifier(num_labels, model_config["dropout"])
    if model_name == "bert":
        return BertTagClassifier(
            model_config["text_model"], num_labels,
            model_config["freeze_bert_layers"], model_config["dropout"]
        )
    if model_name == "gnn":
        return GNNClassifier(**common_gnn)
    fusion_common = dict(
        graph_input_dim=input_dim, num_labels=num_labels, model_name=model_config["text_model"],
        gnn_hidden_dim=model_config["gnn_hidden_dim"], gnn_layers=model_config["gnn_layers"],
        fusion_dim=model_config["fusion_dim"], freeze_layers=model_config["freeze_bert_layers"],
        dropout=model_config["dropout"],
    )
    if model_name == "early_concat":
        return EarlyConcatFusion(**fusion_common)
    if model_name == "cross_attention":
        return CrossAttentionFusion(**fusion_common, attention_heads=model_config["attention_heads"])
    raise ValueError(f"Unknown model: {model_name}")


def make_loaders(model_name: str, config: dict[str, Any], manifest: pd.DataFrame):
    mode = MODEL_MODES[model_name]
    tokenizer = None
    if mode in {"text", "multimodal"}:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model"])
    batch_size = config["training"]["bert_batch_size"] if tokenizer else config["training"]["batch_size"]
    collator = BatchCollator(mode, tokenizer, config["model"]["max_text_length"])
    loaders = {}
    for split in ("training", "validation", "test"):
        dataset = ManifestDataset(manifest, split, mode)
        generator = torch.Generator().manual_seed(config["seed"])
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "training"),
            num_workers=config["training"]["num_workers"], collate_fn=collator,
            pin_memory=torch.cuda.is_available(), generator=generator,
        )
    return loaders, tokenizer


def label_matrix(dataset: ManifestDataset) -> torch.Tensor:
    return torch.stack([dataset[index]["labels"] for index in range(len(dataset))])


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> AdamW:
    bert_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (bert_parameters if "transformer" in name else other_parameters).append(parameter)
    groups = [{"params": other_parameters, "lr": config["training"]["learning_rate"]}]
    if bert_parameters:
        groups.append({"params": bert_parameters, "lr": config["training"]["bert_learning_rate"]})
    return AdamW(groups, weight_decay=config["training"]["weight_decay"])


def run_epoch(model, loader, optimizer, loss_function, device, scaler, training: bool) -> float:
    model.train(training)
    total_loss, examples = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model_forward(model, batch)
            loss = loss_function(output["logits"], batch["labels"])
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        batch_size = batch["labels"].size(0)
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
    return total_loss / max(examples, 1)


def train_experiment(model_name: str, config_path: str, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_name not in MODEL_MODES:
        raise ValueError(f"Trainable model must be one of {sorted(MODEL_MODES)}")
    config = load_config(config_path, profile)
    paths = ensure_project_dirs(config)
    seed_everything(config["seed"])
    manifest = pd.read_csv(paths["manifest"])
    input_dim, num_labels = infer_dimensions(manifest)
    loaders, tokenizer = make_loaders(model_name, config, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, config, input_dim, num_labels).to(device)
    optimizer = build_optimizer(model, config)

    targets = label_matrix(loaders["training"].dataset)
    positive = targets.sum(dim=0)
    negative = targets.size(0) - positive
    pos_weight = (negative / positive.clamp(min=1)).clamp(max=20).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "validation_loss": [], "validation_macro_f1": [], "validation_micro_f1": []}
    best_score, best_state, remaining_patience = -1.0, None, config["training"]["patience"]
    started = time.time()
    for epoch in range(config["training"]["epochs"]):
        train_loss = run_epoch(model, loaders["training"], optimizer, loss_function, device, scaler, True)
        validation_loss = run_epoch(model, loaders["validation"], optimizer, loss_function, device, scaler, False)
        raw_validation = collect_predictions(model, loaders["validation"], device)
        threshold = tune_threshold(raw_validation["targets"], raw_validation["probabilities"])
        validation_metrics = classification_metrics(raw_validation["targets"], raw_validation["probabilities"], threshold)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_macro_f1"].append(validation_metrics["macro_f1"])
        history["validation_micro_f1"].append(validation_metrics["micro_f1"])
        print(
            f"{model_name} epoch {epoch + 1:02d}: train={train_loss:.4f} "
            f"val={validation_loss:.4f} macro-F1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["macro_f1"] > best_score:
            best_score = validation_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())
            remaining_patience = config["training"]["patience"]
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    raw_validation = collect_predictions(model, loaders["validation"], device)
    threshold = tune_threshold(raw_validation["targets"], raw_validation["probabilities"])
    include_details = model_name == "cross_attention"
    raw_test = collect_predictions(model, loaders["test"], device, include_details=include_details)
    metrics = classification_metrics(raw_test["targets"], raw_test["probabilities"], threshold)
    metrics.update(
        {
            "model": model_name, "best_validation_macro_f1": best_score,
            "epochs_trained": len(history["train_loss"]), "duration_seconds": time.time() - started,
            "trainable_parameters": count_parameters(model), "total_parameters": count_parameters(model, False),
            "profile": profile, "test_examples": len(raw_test["targets"]),
        }
    )
    checkpoint_path = paths["checkpoints_dir"] / f"{model_name}.pt"
    torch.save(
        {"model_state": model.state_dict(), "model_name": model_name, "input_dim": input_dim,
         "num_labels": num_labels, "config": config, "threshold": threshold}, checkpoint_path
    )
    save_json(history, paths["results_dir"] / f"{model_name}_history.json")
    save_json(metrics, paths["results_dir"] / f"{model_name}_metrics.json")
    np.savez_compressed(
        paths["results_dir"] / f"{model_name}_test_outputs.npz",
        **{key: value for key, value in raw_test.items() if isinstance(value, np.ndarray)}
    )
    details = {"history": history, "checkpoint": str(checkpoint_path), "raw_test": raw_test, "tokenizer": tokenizer}
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", choices=["smoke", "quick", "final"], default="quick")
    parser.add_argument("--model", choices=sorted(MODEL_MODES), required=True)
    args = parser.parse_args()
    metrics, _ = train_experiment(args.model, args.config, args.profile)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

