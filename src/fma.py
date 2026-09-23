from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEXT_COLUMNS = {
    "track_title": ("track", "title"),
    "album_title": ("album", "title"),
    "artist_name": ("artist", "name"),
}


def load_tracks(metadata_dir: str | Path) -> pd.DataFrame:
    path = Path(metadata_dir) / "tracks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.download_data first.")
    tracks = pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False)
    tracks.index = tracks.index.astype(int)
    return tracks


def load_genres(metadata_dir: str | Path) -> pd.DataFrame:
    path = Path(metadata_dir) / "genres.csv"
    genres = pd.read_csv(path, index_col=0)
    genres.index = genres.index.astype(int)
    return genres


def audio_path(audio_dir: str | Path, track_id: int) -> Path:
    tid = f"{int(track_id):06d}"
    return Path(audio_dir) / tid[:3] / f"{tid}.mp3"


def make_text(row: pd.Series, fields: list[str]) -> str:
    pieces: list[str] = []
    labels = {"track_title": "Track", "album_title": "Album", "artist_name": "Artist"}
    for field in fields:
        column = TEXT_COLUMNS[field]
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            pieces.append(f"{labels[field]}: {str(value).strip()}")
    return ". ".join(pieces) if pieces else "Unknown music track"


def _parse_genre_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


def build_examples(
    tracks: pd.DataFrame,
    genres: pd.DataFrame,
    audio_dir: str | Path,
    label_mode: str,
    text_fields: list[str],
    max_labels: int = 20,
    min_label_frequency: int = 25,
    max_tracks: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    subset = tracks[("set", "subset")].astype(str).str.lower()
    selected = tracks[subset == "small"].copy()
    selected = selected[selected[("set", "split")].isin(["training", "validation", "test"])]

    if label_mode == "top_genre":
        names = sorted(selected[("track", "genre_top")].dropna().astype(str).unique())
        label_map = {name: idx for idx, name in enumerate(names)}
    elif label_mode == "genres_all":
        training = selected[selected[("set", "split")] == "training"]
        counts = Counter(
            genre_id
            for value in training[("track", "genres_all")]
            for genre_id in _parse_genre_ids(value)
        )
        ids = [gid for gid, count in counts.most_common(max_labels) if count >= min_label_frequency]
        label_map = {
            str(genres.loc[gid, "title"] if gid in genres.index else f"genre_{gid}"): idx
            for idx, gid in enumerate(ids)
        }
        genre_id_to_index = {gid: idx for idx, gid in enumerate(ids)}
    else:
        raise ValueError("label_mode must be 'top_genre' or 'genres_all'")

    records: list[dict[str, Any]] = []
    for track_id, row in selected.iterrows():
        path = audio_path(audio_dir, track_id)
        if not path.exists():
            continue
        labels = np.zeros(len(label_map), dtype=np.float32)
        if label_mode == "top_genre":
            name = str(row[("track", "genre_top")])
            if name not in label_map:
                continue
            labels[label_map[name]] = 1.0
        else:
            for gid in _parse_genre_ids(row[("track", "genres_all")]):
                if gid in genre_id_to_index:
                    labels[genre_id_to_index[gid]] = 1.0
            if labels.sum() == 0:
                continue
        records.append(
            {
                "track_id": int(track_id),
                "split": str(row[("set", "split")]),
                "artist_id": str(row.get(("artist", "id"), "")),
                "audio_path": str(path.resolve()),
                "text": make_text(row, text_fields),
                "labels": json.dumps(labels.tolist()),
            }
        )

    examples = pd.DataFrame(records)
    if max_tracks and len(examples) > max_tracks:
        fractions = examples["split"].value_counts(normalize=True)
        sampled = []
        for split, fraction in fractions.items():
            group = examples[examples["split"] == split]
            n = max(1, min(len(group), round(max_tracks * fraction)))
            sampled.append(group.sample(n=n, random_state=seed))
        examples = pd.concat(sampled).sort_values("track_id").head(max_tracks).reset_index(drop=True)
    return examples, label_map

