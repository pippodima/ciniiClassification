"""
figures/fig_lcc_hierarchy.py  —  G6: LCC two-level code illustration.

Pure schematic: needs NO data, runs anywhere:
    python figures/fig_lcc_hierarchy.py --out reports/thesis_figures/fig_lcc_hierarchy.png

Shows main class → subclass → division, with the assigned two-level path
(T → TK → TK7874) highlighted, and a few siblings for context.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DIM  = "#eaeaea"
HOT  = "#fbe2c4"
EDGE = "#888"


def node(ax, cx, cy, w, h, code, name, hot, fs=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.4 if hot else 1.0,
                 edgecolor="#c07a2b" if hot else EDGE,
                 facecolor=HOT if hot else DIM))
    ax.text(cx, cy + 0.12, code, ha="center", va="center", fontsize=fs + 1, fontweight="bold")
    ax.text(cx, cy - 0.16, name, ha="center", va="center", fontsize=fs - 2.5, color="#333")


def link(ax, p0, p1, hot=False):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color="#c07a2b" if hot else EDGE, lw=1.8 if hot else 1.0,
            zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/thesis_figures/fig_lcc_hierarchy.png")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    w, h = 2.7, 0.85

    # level labels
    for y, lab in [(6.6, "main class\n(1 letter)"),
                   (4.0, "subclass\n(2–3 letters)"),
                   (1.4, "division\n(letters + number)")]:
        ax.text(0.5, y, lab, ha="left", va="center", fontsize=8.5,
                color="#777", style="italic")

    # main class
    node(ax, 6.0, 6.6, w, h, "T", "Technology", hot=True)

    # subclasses (TK highlighted)
    subs = [(3.2, "TA", "Civil / Gen. Eng.", False),
            (6.0, "TK", "Electrical & Electronic Eng.", True),
            (8.8, "TJ", "Mechanical Eng.", False)]
    for x, c, nm, hot in subs:
        node(ax, x, 4.0, w, h, c, nm, hot)
        link(ax, (6.0, 6.6 - h / 2), (x, 4.0 + h / 2), hot=hot)

    # divisions under TK (TK7874 highlighted)
    divs = [(4.4, "TK5101", "Telecommunication", False),
            (7.6, "TK7874", "Microelectronics", True)]
    for x, c, nm, hot in divs:
        node(ax, x, 1.4, w, h, c, nm, hot)
        link(ax, (6.0, 4.0 - h / 2), (x, 1.4 + h / 2), hot=hot)

    ax.text(11.4, 7.5, "assigned path\nhighlighted", ha="right", va="top",
            fontsize=8.5, color="#c07a2b", style="italic")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
