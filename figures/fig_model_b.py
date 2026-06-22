"""
figures/fig_model_b.py  —  G2: two-head MLP (Model B) architecture schematic.

Pure schematic: needs NO data, runs anywhere:
    python figures/fig_model_b.py --out reports/thesis_figures/fig_model_b.png

Shows the shared backbone feeding two heads (subclass + division), and the
inference-time hierarchy mask: the predicted subclass gates the division logits so
the two predictions are guaranteed taxonomically consistent.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

IN    = "#e6e6e6"
CORE  = "#d6e6f5"
HEAD  = "#cfe3f7"
MASK  = "#fbe2c4"
OUT   = "#d8efd8"


def box(ax, cx, cy, w, h, title, sub, colour, tfs=10.5, sfs=8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.1, edgecolor="#555", facecolor=colour))
    if sub:
        ax.text(cx, cy + 0.13, title, ha="center", va="center", fontsize=tfs, fontweight="bold")
        ax.text(cx, cy - 0.16, sub, ha="center", va="center", fontsize=sfs, color="#333")
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs, fontweight="bold")


def arrow(ax, p0, p1, style="-|>", color="#555", lw=1.4):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/thesis_figures/fig_model_b.png")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    xL, xR = 3.6, 8.4          # left (subclass) / right (division) columns
    xC = 6.0                   # centre

    # input + backbone (centre column)
    box(ax, xC, 9.2, 5.6, 0.9, "Document embedding",
        "Qwen3-0.6B · 1024-d · L2-normalised", IN)
    box(ax, xC, 7.5, 5.6, 1.1, "Shared backbone",
        "1024 → 512 → 256\nBatchNorm · ReLU · dropout 0.3", CORE)
    arrow(ax, (xC, 8.75), (xC, 8.05))

    # split to two heads
    box(ax, xL, 5.6, 3.4, 0.95, "Subclass head", "256 → 35 logits", HEAD)
    box(ax, xR, 5.6, 3.4, 0.95, "Division head", "256 → 163 logits", HEAD)
    arrow(ax, (xC, 6.95), (xL, 6.08))
    arrow(ax, (xC, 6.95), (xR, 6.08))

    # subclass output (left)
    box(ax, xL, 3.7, 3.4, 0.85, "softmax → subclass", "e.g.  TK", OUT)
    arrow(ax, (xL, 5.12), (xL, 4.12))

    # hierarchy mask + division output (right)
    box(ax, xR, 3.7, 3.6, 1.1, "Hierarchy mask",
        "division logits outside the\npredicted subclass → −∞", MASK)
    arrow(ax, (xR, 5.12), (xR, 4.25))
    box(ax, xR, 1.7, 3.6, 0.9, "softmax → division", "e.g.  TK7874  (consistent)", OUT)
    arrow(ax, (xR, 3.15), (xR, 2.15))

    # gate: predicted subclass conditions the mask
    arrow(ax, (xL + 1.7, 3.7), (xR - 1.8, 3.7), style="-|>", color="#c07a2b", lw=1.6)
    ax.text(xC, 4.05, "gates", ha="center", va="bottom", fontsize=8.5,
            color="#c07a2b", style="italic")

    ax.text(6.0, 0.45,
            "Training: joint loss  0.35·CE(subclass) + 0.65·CE(division)",
            ha="center", va="center", fontsize=9, color="#444")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
