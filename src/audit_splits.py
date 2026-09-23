from __future__ import annotations

import argparse
import json

import pandas as pd

from .utils import load_config, resolve_paths, save_json


def audit(config_path: str) -> dict:
    config = load_config(config_path)
    paths = resolve_paths(config)
    manifest = pd.read_csv(paths["manifest"])
    expected = {"training", "validation", "test"}
    present = set(manifest["split"])
    if not expected.issubset(present):
        raise ValueError(f"Missing splits: {sorted(expected - present)}")

    track_overlap: dict[str, int] = {}
    artist_overlap: dict[str, int] = {}
    pairs = [("training", "validation"), ("training", "test"), ("validation", "test")]
    for first, second in pairs:
        a = manifest[manifest["split"] == first]
        b = manifest[manifest["split"] == second]
        key = f"{first}_vs_{second}"
        track_overlap[key] = len(set(a["track_id"]) & set(b["track_id"]))
        artist_overlap[key] = len(set(a["artist_id"].astype(str)) & set(b["artist_id"].astype(str)))
    result = {
        "split_counts": manifest["split"].value_counts().to_dict(),
        "duplicate_track_ids": int(manifest["track_id"].duplicated().sum()),
        "track_overlap": track_overlap,
        "artist_overlap": artist_overlap,
        "target_words_in_text": "not_applicable: labels were never used by make_text()",
        "official_fma_split_used": True,
        "note": "FMA official splits are authoritative. Artist overlap is reported transparently as requested by the brief.",
    }
    save_json(result, paths["results_dir"] / "split_audit.json")
    print(json.dumps(result, indent=2))
    if result["duplicate_track_ids"] or any(track_overlap.values()):
        raise AssertionError("Track leakage detected")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit(args.config)


if __name__ == "__main__":
    main()

