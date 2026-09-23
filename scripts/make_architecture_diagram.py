from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor="#1f2937", linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#334155", linewidth=1.4))


def main():
    fig, axis = plt.subplots(figsize=(11, 4.1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    blue, purple, green = "#dbeafe", "#ede9fe", "#dcfce7"

    box(axis, 0.03, 0.65, 0.14, 0.17, "Audio clip\n22,050 Hz", blue)
    box(axis, 0.23, 0.65, 0.17, 0.17, "5 s segments\n330 features/node", blue)
    box(axis, 0.46, 0.65, 0.17, 0.17, "Temporal +\nsimilarity graph", blue)
    box(axis, 0.69, 0.65, 0.14, 0.17, "GraphSAGE\nmean pool", blue)
    box(axis, 0.08, 0.18, 0.18, 0.17, "Title + album + artist\n(no target labels)", purple)
    box(axis, 0.34, 0.18, 0.15, 0.17, "DistilBERT\ntoken vectors", purple)
    box(axis, 0.69, 0.18, 0.14, 0.17, "Cross-attention\nFusion z", green)
    box(axis, 0.88, 0.41, 0.10, 0.17, "Genre\nlogits", green)

    arrow(axis, (0.17, 0.735), (0.23, 0.735))
    arrow(axis, (0.40, 0.735), (0.46, 0.735))
    arrow(axis, (0.63, 0.735), (0.69, 0.735))
    arrow(axis, (0.26, 0.265), (0.34, 0.265))
    arrow(axis, (0.49, 0.265), (0.69, 0.265))
    arrow(axis, (0.76, 0.65), (0.76, 0.35))
    arrow(axis, (0.83, 0.265), (0.91, 0.41))
    axis.text(0.58, 0.73, "G", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axis.text(0.58, 0.26, r"$H_{text}$", ha="center", va="bottom", fontsize=9)
    axis.text(0.78, 0.49, "g", ha="left", va="center", fontsize=9, fontweight="bold")
    axis.set_title("Leakage-safe GNN-BERT music context pipeline", fontsize=13, fontweight="bold")
    output = Path(__file__).resolve().parents[1] / "plots" / "architecture_pipeline.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

