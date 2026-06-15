"""
figures/fig_silhouette.py  —  Fig 3 (v3): cluster quality from the v3_300k
HDBSCAN grid search. Plots silhouette vs number of clusters for every
(min_cluster_size, min_samples) config tried, and marks the selected one.
This is the v3 equivalent of the v2 find_optimal_k sweep.

Run on the SERVER (or anywhere training_runs/v3_300k/hdbscan_tuning.json is):
    python figures/fig_silhouette.py \
        --tuning training_runs/v3_300k/hdbscan_tuning.json \
        --out reports/thesis_figures/fig3_silhouette_v3.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="training_runs/v3_300k/hdbscan_tuning.json")
    ap.add_argument("--out", default="reports/thesis_figures/fig3_silhouette_v3.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    data = json.loads(Path(args.tuning).read_text())
    allr = [r for r in data["all"] if r.get("score", 0) > 0 and r.get("n_clusters")]
    best = data["best"]

    xs = [r["n_clusters"] for r in allr]
    ys = [r["score"] for r in allr]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.scatter(xs, ys, s=42, color="#2f6fed", alpha=.75, edgecolors="white",
               linewidths=.5, label="grid-search configs")
    ax.scatter([best["n_clusters"]], [best["score"]], s=170, facecolors="none",
               edgecolors="#2e8b57", lw=2.2, zorder=5, label="selected config")

    # headroom so the top point + annotation don't collide with the title
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.25 or 0.01
    ax.set_ylim(ymin - pad, ymax + pad)
    xmin, xmax = min(xs), max(xs)
    ax.set_xlim(xmin - (xmax - xmin) * .08, xmax + (xmax - xmin) * .12)

    # place the label in open space (lower area) with a leader line to the point
    ax.annotate(
        f"selected: k={best['n_clusters']}, sil={best['score']:.3f}\n"
        f"(mcs={best['mcs']}, ms={best['ms']})",
        xy=(best["n_clusters"], best["score"]),
        xytext=(0.55, 0.18), textcoords="axes fraction",
        color="#2e8b57", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.2))

    ax.set(xlabel="number of clusters", ylabel="silhouette score",
           title="v3_300k HDBSCAN grid search: silhouette vs cluster count")
    ax.legend(loc="lower left", fontsize=9, framealpha=.9)
    ax.grid(alpha=.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}   ({len(allr)} configs, best k={best['n_clusters']})")


if __name__ == "__main__":
    main()
