"""
figures/fig_timeline.py  —  how the corpus composition evolves over time.
Run on the SERVER:
    python figures/fig_timeline.py --input classified/classified_v3_300k.parquet \
        --out-dir reports/thesis_figures

Produces:
    fig23_maintype_over_time.png   stacked-area: LCC main-class SHARE per year
    fig24_main_decade_heatmap.png  main class × decade counts (log-coloured)
Shows shifts in research focus (e.g. growth of medicine / technology).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_YEAR = re.compile(r"(1[5-9]\d{2}|20\d{2})")
MAIN_NAMES = {"Q": "Science", "R": "Medicine", "T": "Technology", "H": "Social Sci.",
              "G": "Geography", "S": "Agriculture", "B": "Phil/Psych", "P": "Lang/Lit",
              "L": "Education", "J": "Pol.Sci"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--out-dir", default="reports/thesis_figures")
    ap.add_argument("--ymin", type=int, default=1970)
    ap.add_argument("--ymax", type=int, default=2022)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    pf = pq.ParquetFile(args.input)
    rows = []
    for b in pf.iter_batches(batch_size=200000, columns=["publication_date", "pred_lcc_main"]):
        d = b.to_pandas()
        yr = d["publication_date"].astype(str).str.extract(_YEAR)[0]
        rows.append(pd.DataFrame({"year": pd.to_numeric(yr, errors="coerce"),
                                  "main": d["pred_lcc_main"].astype(str)}))
    df = pd.concat(rows).dropna()
    df = df[(df.year >= args.ymin) & (df.year <= args.ymax)]
    df["year"] = df["year"].astype(int)
    print(f"  {len(df):,} docs in {args.ymin}-{args.ymax}")

    top = df["main"].value_counts().head(8).index.tolist()
    df["main"] = np.where(df["main"].isin(top), df["main"], "other")
    ct = pd.crosstab(df["year"], df["main"])
    order = [c for c in top if c in ct.columns] + (["other"] if "other" in ct.columns else [])
    ct = ct[order]

    # ── Fig 23 — share over time (stacked area) ──────────────────────────────
    share = ct.div(ct.sum(axis=1), axis=0)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.stackplot(share.index, *[share[c] for c in share.columns],
                 labels=[MAIN_NAMES.get(c, c) for c in share.columns],
                 colors=[cmap(i) for i in range(len(share.columns))], alpha=.9)
    ax.set(xlim=(args.ymin, args.ymax), ylim=(0, 1),
           xlabel="publication year", ylabel="share of papers",
           title="LCC main-class composition over time")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=.9)
    fig.tight_layout(); fig.savefig(out / "fig23_maintype_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig23_maintype_over_time.png")

    # ── Fig 24 — main class × decade heatmap ─────────────────────────────────
    dec = (df["year"] // 10 * 10).astype(int)
    h = pd.crosstab(df["main"], dec).reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(np.log10(h.values + 1), cmap="magma", aspect="auto")
    ax.set_xticks(range(h.shape[1])); ax.set_xticklabels([f"{c}s" for c in h.columns])
    ax.set_yticks(range(h.shape[0]))
    ax.set_yticklabels([MAIN_NAMES.get(c, c) for c in h.index])
    fig.colorbar(im, label="log10(papers + 1)")
    ax.set(title="Papers per LCC main class and decade")
    fig.tight_layout(); fig.savefig(out / "fig24_main_decade_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig24_main_decade_heatmap.png")

    print(f"\n  timeline figures → {out}/")


if __name__ == "__main__":
    main()
