from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from .audio_features import AudioFeatureConfig, extract_track
from .graph_builder import make_graph
from .train import build_model
from .utils import load_json


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["config"],
        checkpoint["input_dim"], checkpoint["num_labels"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(audio_path: str, text: str, checkpoint_path: str, label_map_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    config = checkpoint["config"]
    dc = config["data"]
    feature_config = AudioFeatureConfig(
        sample_rate=dc["sample_rate"], duration_seconds=dc["duration_seconds"],
        segment_seconds=dc["segment_seconds"], n_mels=dc["n_mels"], n_mfcc=dc["n_mfcc"],
        n_fft=dc["n_fft"], hop_length=dc["hop_length"],
    )
    nodes, _ = extract_track(audio_path, feature_config)
    empty_labels = torch.zeros(checkpoint["num_labels"])
    graph = make_graph(
        nodes, empty_labels, 0, text,
        dc["similarity_threshold"], dc["similarity_top_k"]
    )
    graph = Batch.from_data_list([graph]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model"])
    tokens = tokenizer(
        [text], padding="max_length", truncation=True,
        max_length=config["model"]["max_text_length"], return_tensors="pt"
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    output = model(graph=graph, tokens=tokens)
    probabilities = torch.sigmoid(output["logits"])[0].cpu().numpy()
    label_map = load_json(label_map_path)
    names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    top = probabilities.argsort()[-5:][::-1]
    return {
        "decision_threshold": checkpoint["threshold"],
        "predictions": [{"label": names[index], "probability": float(probabilities[index])} for index in top],
        "positive_labels": [names[i] for i, p in enumerate(probabilities) if p >= checkpoint["threshold"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", default="results/checkpoints/cross_attention.pt")
    parser.add_argument("--label-map", default="data/processed/label_map.json")
    args = parser.parse_args()
    print(json.dumps(predict(args.audio, args.text, args.checkpoint, args.label_map), indent=2))


if __name__ == "__main__":
    main()

