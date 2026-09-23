from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import classification_metrics, tune_threshold
from .train import train_experiment
from .train_contrastive import train_contrastive
from .utils import ensure_project_dirs, load_config, load_json, save_json, seed_everything
from .visualize import create_case_studies, plot_model_comparison, plot_training_history, plot_tsne


def random_baseline(manifest: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    parse = lambda series: np.stack(series.map(json.loads).to_list()).astype(np.float32)
    train = parse(manifest[manifest["split"] == "training"]["labels"])
    validation = parse(manifest[manifest["split"] == "validation"]["labels"])
    test = parse(manifest[manifest["split"] == "test"]["labels"])
    prevalence = train.mean(axis=0)
    validation_probs = rng.random(validation.shape) * 0.5 + prevalence[None, :] * 0.5
    threshold = tune_threshold(validation, validation_probs)
    test_probs = rng.random(test.shape) * 0.5 + prevalence[None, :] * 0.5
    metrics = classification_metrics(test, test_probs, threshold)
    metrics.update({"model": "random", "seed": seed, "test_examples": len(test)})
    return metrics


def run_all(config_path: str, profile: str, include_bonus: bool = False) -> dict:
    config = load_config(config_path, profile)
    paths = ensure_project_dirs(config)
    seed_everything(config["seed"])
    manifest = pd.read_csv(paths["manifest"])
    label_map = load_json(paths["label_map"])
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    results: dict[str, dict] = {"random": random_baseline(manifest, config["seed"])}
    save_json(results["random"], paths["results_dir"] / "random_metrics.json")
    for model_name in ["cnn", "bert", "gnn", "early_concat", "cross_attention"]:
        metrics, details = train_experiment(model_name, config_path, profile)
        results[model_name] = metrics
        plot_training_history(details["history"], model_name, paths["plots_dir"] / f"{model_name}_training_curves.png")
        if model_name == "cross_attention":
            raw = details["raw_test"]
            if "embeddings" in raw:
                plot_tsne(raw["embeddings"], raw["targets"], label_names, paths["plots_dir"] / "fusion_tsne.png", config["seed"])
            create_case_studies(
                raw, details["tokenizer"], manifest, label_names, paths["plots_dir"],
                paths["results_dir"] / "case_studies.json", count=3
            )
    save_json(results, paths["results_dir"] / "metrics.json")
    plot_model_comparison(results, paths["plots_dir"] / "model_comparison.png")
    if include_bonus:
        results["contrastive_retrieval"] = train_contrastive(config_path, profile)
        save_json(results, paths["results_dir"] / "metrics.json")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", choices=["smoke", "quick", "final"], default="quick")
    parser.add_argument("--include-bonus", action="store_true")
    args = parser.parse_args()
    run_all(args.config, args.profile, args.include_bonus)


if __name__ == "__main__":
    main()
