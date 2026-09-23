from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from .utils import save_json


STYLE = {"figure.facecolor": "white", "axes.facecolor": "#f8fafc", "axes.grid": True, "grid.alpha": 0.2}


def plot_training_history(history: dict[str, list[float]], model_name: str, destination: str | Path) -> None:
    plt.rcParams.update(STYLE)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set(title=f"{model_name}: BCE loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["validation_macro_f1"], marker="o", label="Macro-F1")
    axes[1].plot(epochs, history["validation_micro_f1"], marker="o", label="Micro-F1")
    axes[1].set(title=f"{model_name}: validation F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: dict[str, dict[str, Any]], destination: str | Path) -> None:
    rows = [
        {"Model": name, "Macro-F1": values["macro_f1"], "Micro-F1": values["micro_f1"], "AUC-PR": values["macro_auc_pr"]}
        for name, values in metrics.items()
    ]
    frame = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(data=frame, x="Model", y="Score", hue="Metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("Fair test-set comparison on the same FMA split")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(
    embeddings: np.ndarray, targets: np.ndarray, label_names: list[str], destination: str | Path, seed: int = 42
) -> None:
    if len(embeddings) < 5:
        return
    perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
    coordinates = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(embeddings)
    primary = targets.argmax(axis=1)
    fig, axis = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=len(label_names))
    for index, name in enumerate(label_names):
        mask = primary == index
        if mask.any():
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=24, alpha=0.72, label=name, color=palette[index % len(palette)])
    axis.set(title="t-SNE of cross-attention fusion embeddings", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_case_studies(
    raw: dict[str, Any], tokenizer, manifest: pd.DataFrame, label_names: list[str],
    destination_dir: str | Path, results_path: str | Path, count: int = 3,
) -> list[dict[str, Any]]:
    if "attention" not in raw:
        return []
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    confidence = raw["probabilities"].max(axis=1)
    chosen = np.argsort(confidence)[-count:][::-1]
    studies: list[dict[str, Any]] = []
    manifest_by_id = manifest.set_index("track_id")
    for rank, index in enumerate(chosen, 1):
        track_id = int(raw["track_ids"][index])
        graph = torch.load(manifest_by_id.loc[track_id, "graph_path"], map_location="cpu", weights_only=False)
        probabilities = raw["probabilities"][index]
        target = raw["targets"][index]
        top_predictions = probabilities.argsort()[-3:][::-1]
        tokens = tokenizer.convert_ids_to_tokens(raw["token_ids"][index].tolist())
        weights = raw["attention"][index]
        valid = [(token, float(weight)) for token, weight in zip(tokens, weights) if token not in tokenizer.all_special_tokens]
        top_tokens = sorted(valid, key=lambda pair: pair[1], reverse=True)[:8]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for source, target_node in graph.edge_index.T.tolist():
            axes[0].plot([source, target_node], [0, 0], color="#94a3b8", alpha=0.45, linewidth=1)
        axes[0].scatter(range(graph.num_nodes), np.zeros(graph.num_nodes), s=180, c=np.arange(graph.num_nodes), cmap="viridis", zorder=3)
        axes[0].set(title=f"Track {track_id}: segment graph", xlabel="5-second segment index")
        axes[0].set_yticks([])
        names = [label_names[i] for i in top_predictions]
        axes[1].barh(names[::-1], probabilities[top_predictions][::-1], color="#2563eb")
        axes[1].set(xlim=(0, 1), xlabel="Probability", title="Top predictions")
        if top_tokens:
            token_names = [token for token, _ in top_tokens][::-1]
            token_weights = [weight for _, weight in top_tokens][::-1]
            axes[2].barh(token_names, token_weights, color="#7c3aed")
        axes[2].set(xlabel="Mean attention", title="Graph-aligned text tokens")
        fig.suptitle(str(manifest_by_id.loc[track_id, "text"])[:120], fontsize=10)
        fig.tight_layout()
        output = destination_dir / f"case_study_{rank}.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        studies.append(
            {
                "track_id": track_id,
                "text": str(manifest_by_id.loc[track_id, "text"]),
                "true_labels": [name for name, value in zip(label_names, target) if value > 0.5],
                "top_predictions": [{"label": label_names[i], "probability": float(probabilities[i])} for i in top_predictions],
                "highest_attention_tokens": [{"token": token, "weight": weight} for token, weight in top_tokens],
                "graph_path_interpretation": "Edges show temporal or high-cosine-similarity connections among 5-second segments.",
            }
        )
    save_json(studies, results_path)
    return studies
