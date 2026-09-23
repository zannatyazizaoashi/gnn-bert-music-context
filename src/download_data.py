from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import ensure_project_dirs, load_config


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(temporary, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=destination.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    temporary.replace(destination)
    return destination


def extract_zip(archive: Path, raw_dir: Path) -> None:
    expected = raw_dir / archive.stem
    completion_marker = expected / ".extract_complete"
    if completion_marker.exists():
        print(f"Already extracted: {expected}")
        return
    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(raw_dir)
    completion_marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", choices=["metadata", "fma_small"], default="fma_small")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_project_dirs(config)
    raw_dir = paths["raw_dir"]

    metadata_zip = download(config["download"]["metadata_url"], raw_dir / "fma_metadata.zip")
    extract_zip(metadata_zip, raw_dir)
    if args.dataset == "fma_small":
        audio_zip = download(config["download"]["fma_small_url"], raw_dir / "fma_small.zip")
        extract_zip(audio_zip, raw_dir)
    print("Dataset preparation finished.")


if __name__ == "__main__":
    main()

