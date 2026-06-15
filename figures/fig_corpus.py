"""
figures/fig_corpus.py
=====================
Data-dependent thesis figures computed from the FULL classified corpus
(run on the server where the parquet lives — these read all 3.6M rows).

  fig5_lcc_distribution.png   exact top-N LCC subclass distribution
  fig5b_main_distribution.png LCC main-class distribution (all classes)
  fig10_year_distribution.png publication-year histogram

Usage:
    python figures/fig_corpus.py \
        --input classified/classified_v3_300k.parquet \
        --out-dir reports/thesis_figures --top 20
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "search"))
try:
    from lcc_names import sub_label, main_label  # reuse the search name map
except Exception:
    def sub_label(c): return c
    def main_label(c): return c

ACC = "#2f6fed"
_YEAR = re.compile(r"(1[5-9]\d{2}|20\d{2})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--out-dir", default="reports/thesis_figures")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    # stream just the columns we need (counts only — RAM-safe)
    pf = pq.ParquetFile(args.input)
    sub = pd.Series(dtype="int64")
    main = pd.Series(dtype="int64")
    years = []
    total = 0
    for b in pf.iter_batches(batch_size=100000,
                             columns=["pred_lcc", "pred_lcc_main", "publication_date"]):
        d = b.to_pandas()
        total += len(d)
        sub = sub.add(d["pred_lcc"].value_counts(), fill_value=0)
        main = main.add(d["pred_lcc_main"].value_counts(), fill_value=0)
        yr = d["publication_date"].astype(str).str.extract(_YEAR)[0].dropna().astype(int)
        years.append(yr[(yr >= 1950) & (yr <= 2025)])
    years = pd.concat(years) if years else pd.Series(dtype=int)
    print(f"  scanned {total:,} docs")

    # ── Fig 5 — top subclasses ───────────────────────────────────────────────
    s = sub.sort_values(ascending=False).head(args.top)
    pct = 100 * s / total
    fig, ax = plt.subplots(figsize=(8, max(4.5, .35 * len(s))))
    y = range(len(s))
    ax.barh(list(y), pct.values, color=ACC)
    ax.set_yticks(list(y)); ax.set_yticklabels([sub_label(c) for c in s.index])
    ax.invert_yaxis()
    for i, p in enumerate(pct.values):
        ax.text(p + .15, i, f"{p:.1f}%", va="center", fontsize=8, color="#444")
    ax.set(xlabel="share of corpus (%)",
           title=f"Top {len(s)} LCC subclasses across {total:,} documents")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig5_lcc_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig5_lcc_distribution.png")

    # ── Fig 5b — main classes ────────────────────────────────────────────────
    m = main.sort_values(ascending=False)
    pct = 100 * m / total
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([main_label(c).split(" — ")[0] for c in m.index], pct.values, color=ACC)
    for i, p in enumerate(pct.values):
        if p > 0.5:
            ax.text(i, p + .4, f"{p:.0f}%", ha="center", fontsize=8)
    ax.set(ylabel="share of corpus (%)", title="LCC main-class distribution")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig5b_main_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig5b_main_distribution.png")

    # ── Fig 10 — publication year ────────────────────────────────────────────
    if len(years):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2),
                color=ACC, alpha=.85)
        ax.set(xlabel="publication year", ylabel="documents",
               title=f"Publication-year distribution (median {int(years.median())})")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "fig10_year_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig); print("  ✅ fig10_year_distribution.png")

    print(f"\n  figures → {out}/")


if __name__ == "__main__":
    main()
