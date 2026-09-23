from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from .utils import load_config, load_json, resolve_paths


DISPLAY = {
    "random": "Random/prevalence", "cnn": "CNN mel-spectrogram", "bert": "BERT-only",
    "gnn": "GraphSAGE-only", "early_concat": "Early concatenation", "cross_attention": "Cross-attention fusion",
}


def make_tables(metrics: dict, report_dir: Path, results_dir: Path) -> None:
    models = [name for name in DISPLAY if name in metrics]
    latex = [
        r"\begin{tabular}{lccc}", r"\toprule", r"Model & Macro-F1 & Micro-F1 & AUC-PR \\", r"\midrule"
    ]
    markdown = ["| Model | Macro-F1 | Micro-F1 | AUC-PR |", "|---|---:|---:|---:|"]
    for name in models:
        row = metrics[name]
        label = DISPLAY[name]
        latex.append(f"{label} & {row['macro_f1']:.3f} & {row['micro_f1']:.3f} & {row['macro_auc_pr']:.3f} \\\\")
        markdown.append(f"| {label} | {row['macro_f1']:.3f} | {row['micro_f1']:.3f} | {row['macro_auc_pr']:.3f} |")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (report_dir / "generated_results_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (results_dir / "evaluation_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def package(root: Path, destination: Path) -> None:
    exclude_parts = {"raw", "graphs", "mels", "__pycache__", ".pytest_cache", ".git"}
    exclude_suffixes = {".aux", ".bbl", ".blg", ".log", ".out"}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_dir() or any(part in exclude_parts for part in path.relative_to(root).parts):
                continue
            if path == destination or path.suffix == ".zip":
                continue
            if path.suffix in exclude_suffixes:
                continue
            archive.write(path, path.relative_to(root.parent))


def export(config_path: str) -> Path:
    root = Path(config_path).resolve().parent
    config = load_config(config_path)
    paths = resolve_paths(config, root)
    metrics = load_json(paths["results_dir"] / "metrics.json")
    if metrics.get("status") == "not_run":
        raise RuntimeError("No real metrics found. Run python -m src.run_all before export.")
    make_tables(metrics, root / "report", paths["results_dir"])
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=root / "report", check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        shutil.copy2(root / "report" / "main.pdf", root / "report" / "final_report.pdf")
    graph_samples = list(paths["graph_sample_dir"].glob("*.json"))
    if len(graph_samples) < 20:
        raise RuntimeError(f"Only {len(graph_samples)} graph samples found; at least 20 are required.")
    required = [
        paths["plots_dir"] / "model_comparison.png",
        paths["plots_dir"] / "fusion_tsne.png",
        paths["results_dir"] / "case_studies.json",
        root / "notebooks" / "demo_context.ipynb",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required outputs:\n" + "\n".join(missing))
    destination = root / "submission_bundle.zip"
    package(root, destination)
    print(f"Created {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    export(args.config)


if __name__ == "__main__":
    main()
