"""
figures/fig_pipeline.py  —  G1: end-to-end pipeline architecture diagram.

Pure schematic: needs NO data, runs anywhere (laptop or server):
    python figures/fig_pipeline.py --out reports/thesis_figures/fig_pipeline.png

A single vertical flow from the raw dump to the search interface. Automated stages
are blue, the one manual step is highlighted in orange, data stores are grey, and the
final product is green. Side brackets group the training (300k sample) and inference
(full 3.6M corpus) phases.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA   = "#e6e6e6"
AUTO   = "#d6e6f5"
HUMAN  = "#fbe2c4"
OUT    = "#d8efd8"

# (title, subtitle, colour)
STAGES = [
    ("Raw CiNii dump",            "71,511,821 RDF/XML records",                       DATA),
    ("01 · Parallel parsing",     "JATS/HTML stripped  →  structured Parquet",        AUTO),
    ("02 · Clean & filter",       "abstract + language + CJK guard + doc-type\n"
                                  "→ 3,602,151 English scientific papers",            AUTO),
    ("03 · Embed (Qwen3-0.6B)",   "1024-d, L2-normalised, 768-token cap\n"
                                  "sharded over the full corpus",                     AUTO),
    ("UMAP  →  15 dimensions",    "on a 298,041-document sample",                     AUTO),
    ("HDBSCAN clustering",        "166 density-based clusters (~50% outliers)",        AUTO),
    ("Manual cluster → LCC map",  "only human step  —  35 subclasses / 163 divisions", HUMAN),
    ("Train two-head MLP (Model B)", "147,857 labelled docs · hierarchy mask",        AUTO),
    ("11 · Classify full corpus", "Model B over 3.6M docs  +  centroid trust score",  AUTO),
    ("LCC-faceted Meilisearch",   "subject browsing with trust filter",              OUT),
]

# index ranges (inclusive) for the phase brackets
TRAIN_RANGE = (4, 7)
INFER_RANGE = (8, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/thesis_figures/fig_pipeline.png")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})

    n = len(STAGES)
    bw, bh = 6.4, 0.78          # box width / height (data coords)
    gap = 0.45                  # vertical gap between boxes
    cx = 5.0                    # horizontal centre

    fig, ax = plt.subplots(figsize=(8.4, 1.02 * n))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, n * (bh + gap))
    ax.axis("off")

    centres = []
    for i, (title, sub, colour) in enumerate(STAGES):
        cy = (n - 1 - i) * (bh + gap) + bh / 2
        centres.append(cy)
        box = FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.1, edgecolor="#555", facecolor=colour)
        ax.add_patch(box)
        ax.text(cx, cy + 0.12, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                fontsize=8, color="#333")
        if colour == HUMAN:                      # star marker for the one manual step
            ax.plot(cx - bw / 2 + 0.32, cy + 0.12, marker="*",
                    markersize=13, color="#c07a2b", clip_on=False)

    # arrows between consecutive boxes
    for i in range(n - 1):
        y0 = centres[i] - bh / 2
        y1 = centres[i + 1] + bh / 2
        ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))

    # phase brackets on the right
    def bracket(rng, label):
        top = centres[rng[0]] + bh / 2
        bot = centres[rng[1]] - bh / 2
        x = cx + bw / 2 + 0.35
        ax.plot([x, x + 0.18, x + 0.18, x], [bot, bot, top, top],
                color="#888", lw=1.1)
        ax.text(x + 0.30, (top + bot) / 2, label, rotation=90,
                ha="left", va="center", fontsize=8.5, color="#555")

    bracket(TRAIN_RANGE, "training  (300k sample)")
    bracket(INFER_RANGE, "inference  (full 3.6M)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
