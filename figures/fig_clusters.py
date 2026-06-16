"""
figures/fig_clusters.py  —  cluster-structure figures from the v3_300k clustering.
Run on the SERVER:
    python figures/fig_clusters.py \
        --clusters training_runs/v3_300k/clusters.parquet \
        --out-dir reports/thesis_figures

Produces:
    fig25_cluster_size_dist.png   rank-size of HDBSCAN clusters (log-y)
    fig26_outlier_donut.png       in-cluster vs outlier share (the 50% story)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC = "#2f6fed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="training_runs/v3_300k/clusters.parquet")
    ap.add_argument("--out-dir", default="reports/thesis_figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    cl = pd.read_parquet(args.clusters, columns=["cluster_id"])["cluster_id"].astype(int)
    n_total = len(cl)
    n_out = int((cl < 0).sum())
    sizes = cl[cl >= 0].value_counts().sort_values(ascending=False)
    print(f"  {n_total:,} docs, {len(sizes)} clusters, {n_out:,} outliers")

    # ── Fig 25 — rank-size distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(range(len(sizes)), sizes.values, color=ACC, width=1.0)
    ax.set(yscale="log", xlabel="cluster (rank by size)", ylabel="documents (log)",
           title=f"HDBSCAN cluster sizes ({len(sizes)} clusters; "
                 f"median {int(sizes.median())}, max {int(sizes.max()):,})")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig25_cluster_size_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig25_cluster_size_dist.png")

    # ── Fig 26 — outlier donut ───────────────────────────────────────────────
    inc = n_total - n_out
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.pie([inc, n_out],
           labels=[f"in a cluster\n{inc/n_total:.0%}", f"outlier\n{n_out/n_total:.0%}"],
           colors=[ACC, "#dde1e6"], startangle=90,
           wedgeprops=dict(width=.42, edgecolor="white"), textprops=dict(fontsize=11))
    ax.set(title=f"HDBSCAN coverage of the {n_total:,}-doc sample")
    fig.tight_layout(); fig.savefig(out / "fig26_outlier_donut.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig26_outlier_donut.png")

    print(f"\n  cluster figures → {out}/")


if __name__ == "__main__":
    main()
