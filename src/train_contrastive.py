from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW

from .contrastive import ContrastiveGraphTextModel, retrieval_at_k
from .evaluate import _to_device
from .train import make_loaders
from .datasets import infer_dimensions
from .utils import ensure_project_dirs, load_config, save_json, seed_everything


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    graph_embeddings, text_embeddings, track_ids = [], [], []
    texts = []
    for batch in loader:
        texts.extend(batch["text"])
        batch = _to_device(batch, device)
        output = model(graph=batch["graph"], tokens=batch["tokens"])
        graph_embeddings.append(output["graph_embedding"].cpu())
        text_embeddings.append(output["text_embedding"].cpu())
        track_ids.append(batch["track_id"].cpu())
    return torch.cat(graph_embeddings), torch.cat(text_embeddings), torch.cat(track_ids), texts


def qualitative_retrieval(graph_embeddings, text_embeddings, track_ids, texts, count=10):
    scores = torch.nn.functional.normalize(text_embeddings, dim=-1) @ torch.nn.functional.normalize(graph_embeddings, dim=-1).T
    ranking = scores.argsort(dim=1, descending=True)
    examples = []
    for query_index in range(min(count, len(texts))):
        matches = []
        for match_index in ranking[query_index, :3].tolist():
            matches.append({
                "rank": len(matches) + 1,
                "track_id": int(track_ids[match_index]),
                "similarity": float(scores[query_index, match_index]),
                "matched_text": texts[match_index],
            })
        examples.append({
            "query_track_id": int(track_ids[query_index]),
            "query_text": texts[query_index],
            "top_3_matches": matches,
        })
    return examples


def train_contrastive(config_path: str, profile: str) -> dict:
    config = load_config(config_path, profile)
    paths = ensure_project_dirs(config)
    seed_everything(config["seed"])
    manifest = pd.read_csv(paths["manifest"])
    input_dim, _ = infer_dimensions(manifest)
    loaders, _ = make_loaders("cross_attention", config, manifest)
    mc = config["model"]
    model = ContrastiveGraphTextModel(
        graph_input_dim=input_dim, model_name=mc["text_model"],
        gnn_hidden_dim=mc["gnn_hidden_dim"], gnn_layers=mc["gnn_layers"],
        projection_dim=mc["projection_dim"], freeze_layers=mc["freeze_bert_layers"],
        dropout=mc["dropout"], temperature=mc["temperature"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config["training"]["bert_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = []
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total, examples = 0.0, 0
        for batch in loaders["training"]:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                output = model(graph=batch["graph"], tokens=batch["tokens"])
                loss = model.loss(output)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch["labels"].size(0)
            examples += batch["labels"].size(0)
        graph_val, text_val, _, _ = collect_embeddings(model, loaders["validation"], device)
        validation = retrieval_at_k(graph_val, text_val)
        row = {"epoch": epoch + 1, "train_loss": total / max(examples, 1), **validation}
        history.append(row)
        print(row)
    graph_test, text_test, track_ids, texts = collect_embeddings(model, loaders["test"], device)
    metrics = retrieval_at_k(graph_test, text_test)
    metrics.update({"model": "contrastive", "profile": profile, "test_examples": len(track_ids)})
    torch.save(
        {"model_state": model.state_dict(), "input_dim": input_dim, "config": config},
        paths["checkpoints_dir"] / "contrastive.pt",
    )
    save_json(history, paths["results_dir"] / "contrastive_history.json")
    save_json(metrics, paths["results_dir"] / "retrieval_metrics.json")
    save_json(
        qualitative_retrieval(graph_test, text_test, track_ids, texts),
        Path("retrieval_examples") / "top3_matches.json",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", choices=["smoke", "quick", "final"], default="quick")
    args = parser.parse_args()
    print(json.dumps(train_contrastive(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
