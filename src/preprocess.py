from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .audio_features import AudioFeatureConfig, extract_track
from .fma import build_examples, load_genres, load_tracks
from .graph_builder import graph_to_json, make_graph
from .utils import ensure_project_dirs, load_config, save_json, seed_everything


def preprocess(config_path: str, profile: str) -> pd.DataFrame:
    config = load_config(config_path, profile)
    paths = ensure_project_dirs(config)
    seed_everything(config["seed"])
    data_config = config["data"]
    feature_config = AudioFeatureConfig(
        sample_rate=data_config["sample_rate"],
        duration_seconds=data_config["duration_seconds"],
        segment_seconds=data_config["segment_seconds"],
        n_mels=data_config["n_mels"],
        n_mfcc=data_config["n_mfcc"],
        n_fft=data_config["n_fft"],
        hop_length=data_config["hop_length"],
    )

    tracks = load_tracks(paths["metadata_dir"])
    genres = load_genres(paths["metadata_dir"])
    examples, label_map = build_examples(
        tracks=tracks,
        genres=genres,
        audio_dir=paths["audio_dir"],
        label_mode=data_config["label_mode"],
        text_fields=data_config["text_fields"],
        max_labels=data_config["max_labels"],
        min_label_frequency=data_config["min_label_frequency"],
        max_tracks=data_config.get("max_tracks"),
        seed=config["seed"],
    )
    save_json(label_map, paths["label_map"])
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    completed: list[dict] = []
    failures: list[dict] = []
    for row in tqdm(examples.to_dict("records"), desc="Extracting FMA tracks"):
        track_id = int(row["track_id"])
        graph_path = paths["graph_dir"] / f"{track_id:06d}.pt"
        mel_path = paths["mel_dir"] / f"{track_id:06d}.pt"
        try:
            if graph_path.exists() and mel_path.exists():
                graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            else:
                nodes, mel = extract_track(row["audio_path"], feature_config)
                labels = torch.tensor(json.loads(row["labels"]), dtype=torch.float32)
                graph = make_graph(
                    nodes, labels, track_id, row["text"],
                    data_config["similarity_threshold"], data_config["similarity_top_k"]
                )
                torch.save(graph, graph_path)
                torch.save({"mel": mel, "y": labels, "track_id": track_id}, mel_path)
            item = dict(row)
            item["graph_path"] = str(graph_path.resolve())
            item["mel_path"] = str(mel_path.resolve())
            item["num_nodes"] = int(graph.num_nodes)
            completed.append(item)
        except Exception as exc:  # corrupted/very short FMA files are documented upstream
            failures.append({"track_id": track_id, "error": repr(exc), "trace": traceback.format_exc(limit=1)})

    manifest = pd.DataFrame(completed)
    if manifest.empty:
        raise RuntimeError("No tracks were processed. Check the data paths and downloaded archives.")
    manifest.to_csv(paths["manifest"], index=False)
    save_json(failures, paths["processed_dir"] / "preprocessing_failures.json")

    sample_rows = manifest.sort_values("track_id").head(20)
    for _, row in sample_rows.iterrows():
        graph = torch.load(row["graph_path"], map_location="cpu", weights_only=False)
        save_json(graph_to_json(graph, label_names), paths["graph_sample_dir"] / f"graph_{int(row['track_id']):06d}.json")

    summary = {
        "profile": profile,
        "processed_tracks": len(manifest),
        "failed_tracks": len(failures),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "num_labels": len(label_map),
        "labels": label_names,
        "node_feature_dimension": int(torch.load(manifest.iloc[0]["graph_path"], map_location="cpu", weights_only=False).num_node_features),
    }
    save_json(summary, paths["processed_dir"] / "preprocessing_summary.json")
    print(json.dumps(summary, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", choices=["smoke", "quick", "final"], default="quick")
    args = parser.parse_args()
    preprocess(args.config, args.profile)


if __name__ == "__main__":
    main()

