"""
figures/fig_umap.py  —  2-D UMAP projections of the corpus (Fig 2 family).
Run on the SERVER (needs the embeddings in training_dataset.parquet):

    python figures/fig_umap.py \
        --source training_runs/v3_300k/training_dataset.parquet \
        --out-dir reports/thesis_figures --sample 40000

Computes ONE 2-D UMAP and renders three colourings:
    fig2_umap_main.png     by LCC main class (broad separation)
    fig2_umap_sub.png      by LCC subclass   (every topic group; top-20 coloured)
    fig2_umap_cluster.png  by HDBSCAN cluster id (every cluster as a distinct blob)

The clustering UMAP was 15-D and unsaved, so this recomputes a fresh 2-D one
purely for visualisation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "search"))
try:
    from lcc_names import sub_label, main_label
except Exception:
    def sub_label(c): return c
    def main_label(c): return c

MAIN_NAMES = {"Q": "Q Science", "R": "R Medicine", "T": "T Technology",
              "H": "H Social Sci.", "G": "G Geography", "S": "S Agriculture",
              "B": "B Phil/Psych", "P": "P Lang/Lit", "L": "L Education", "J": "J Pol.Sci"}


def _scatter_categorical(xy, labels, top_n, cmap_name, title, out, namer):
    top = pd.Series(labels).value_counts().head(top_n).index.tolist()
    cmap = plt.get_cmap(cmap_name, max(top_n, len(top)))
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    rest = ~pd.Series(labels).isin(top).values
    ax.scatter(xy[rest, 0], xy[rest, 1], s=3, c="#dde1e6", alpha=.5, linewidths=0,
               label="other", rasterized=True)
    for i, c in enumerate(top):
        m = (labels == c)
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i), alpha=.65,
                   linewidths=0, label=namer(c), rasterized=True)
    ax.set(xticks=[], yticks=[], title=title)
    ax.legend(markerscale=4, loc="upper right", framealpha=.92, fontsize=8, ncol=1)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  ✅ {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_runs/v3_300k/training_dataset.parquet")
    ap.add_argument("--out-dir", default="reports/thesis_figures")
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    import umap

    df = pd.read_parquet(args.source)
    df = df[[c for c in ["embeddings", "lcc", "cluster_id"] if c in df.columns]]
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    X = np.vstack(df["embeddings"].values).astype(np.float32)

    print(f"  UMAP 2D on {len(X):,} points ...")
    xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="cosine", random_state=args.seed).fit_transform(X)

    # by main class
    main = df["lcc"].astype(str).str[0].values
    _scatter_categorical(xy, main, 8, "tab10",
                         "UMAP of the corpus — coloured by LCC main class",
                         out / "fig2_umap_main.png",
                         lambda c: MAIN_NAMES.get(c, c))

    # by subclass
    sub = df["lcc"].astype(str).values
    _scatter_categorical(xy, sub, 20, "tab20",
                         "UMAP of the corpus — coloured by LCC subclass (top 20)",
                         out / "fig2_umap_sub.png", sub_label)

    # by HDBSCAN cluster id (no legend — too many; shows blob structure)
    if "cluster_id" in df.columns:
        cl = df["cluster_id"].astype(int).values
        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        noise = cl < 0
        ax.scatter(xy[noise, 0], xy[noise, 1], s=3, c="#dde1e6", alpha=.4,
                   linewidths=0, rasterized=True)
        n_cl = len(set(cl[~noise]))
        ax.scatter(xy[~noise, 0], xy[~noise, 1], s=4, c=cl[~noise],
                   cmap="gist_ncar", alpha=.7, linewidths=0, rasterized=True)
        ax.set(xticks=[], yticks=[],
               title=f"UMAP of the corpus — coloured by HDBSCAN cluster ({n_cl} clusters)")
        fig.tight_layout(); fig.savefig(out / "fig2_umap_cluster.png", dpi=300, bbox_inches="tight")
        plt.close(fig); print("  ✅ fig2_umap_cluster.png")

    print(f"\n  UMAP figures → {out}/")


if __name__ == "__main__":
    main()
