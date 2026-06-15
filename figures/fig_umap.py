"""
figures/fig_umap.py  —  Fig 2: 2D UMAP scatter coloured by LCC main class.
Run on the SERVER (needs the embeddings in training_dataset.parquet):

    python figures/fig_umap.py \
        --source training_runs/v3_300k/training_dataset.parquet \
        --out reports/thesis_figures/fig2_umap.png --sample 40000

The clustering UMAP was 15-D (not saved), so this recomputes a fresh 2-D UMAP
purely for visualisation. Colour = predicted LCC main class (top classes only;
rest greyed), which shows the clusters are visually coherent and separable.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_runs/v3_300k/training_dataset.parquet")
    ap.add_argument("--out", default="reports/thesis_figures/fig2_umap.png")
    ap.add_argument("--sample", type=int, default=40000, help="points to plot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import umap

    df = pd.read_parquet(args.source, columns=["embeddings", "lcc"])
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    X = np.vstack(df["embeddings"].values).astype(np.float32)
    main = df["lcc"].astype(str).str[0]            # main class letter

    print(f"  UMAP 2D on {len(X):,} points ...")
    xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="cosine", random_state=args.seed).fit_transform(X)

    top = main.value_counts().head(8).index.tolist()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    rest = ~main.isin(top)
    ax.scatter(xy[rest, 0], xy[rest, 1], s=3, c="#d9dde2", alpha=.5, linewidths=0, label="other")
    NAMES = {"Q": "Q Science", "R": "R Medicine", "T": "T Technology",
             "H": "H Social Sci.", "G": "G Geography", "S": "S Agriculture",
             "B": "B Phil/Psych", "P": "P Lang/Lit", "L": "L Education", "J": "J Pol.Sci"}
    for i, c in enumerate(top):
        m = main == c
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i), alpha=.6,
                   linewidths=0, label=NAMES.get(c, c))
    ax.set(xticks=[], yticks=[], title="UMAP projection of the corpus, coloured by LCC main class")
    lgd = ax.legend(markerscale=4, loc="upper right", framealpha=.9, fontsize=9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
