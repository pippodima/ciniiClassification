"""
figures/fig_confusion.py  —  subclass confusion heatmap (journal-gold vs predicted).
Run on the SERVER:
    python figures/fig_confusion.py \
        --input classified/classified_v3_300k.parquet \
        --gold  reports/journal_validation_v3_300k/journal_gold_TK.csv \
        --out   reports/thesis_figures/fig27_confusion.png

Row-normalised P(pred | journal-gold) over the most frequent gold subclasses.
The diagonal = correct; bright off-diagonal cells reveal systematic, usually
adjacent confusions (e.g. QP→RC physiology/medicine, TA→TK materials/electronics).
Uses the single-label TK gold so labels are atomic.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _clean(t):  # must match 16/17 journal normalisation
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--gold", default="reports/journal_validation_v3_300k/journal_gold_TK.csv")
    ap.add_argument("--out", default="reports/thesis_figures/fig27_confusion.png")
    ap.add_argument("--top", type=int, default=18, help="most frequent gold subclasses")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    gold = pd.read_csv(args.gold, dtype=str).fillna("")
    gold["gold_lcc_sub"] = gold["gold_lcc_sub"].str.strip().str.upper()
    gold = gold[gold["gold_lcc_sub"] != ""]
    gmap = dict(zip(gold["journal"].map(_clean), gold["gold_lcc_sub"]))

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.input)
    parts = []
    for b in pf.iter_batches(batch_size=200000, columns=["journal", "pred_lcc"]):
        d = b.to_pandas()
        g = d["journal"].astype(str).map(_clean).map(gmap)
        m = g.notna()
        parts.append(pd.DataFrame({"gold": g[m].values, "pred": d["pred_lcc"][m].values}))
    df = pd.concat(parts)
    print(f"  {len(df):,} papers matched to gold journals")

    classes = df["gold"].value_counts().head(args.top).index.tolist()
    sub = df[df["gold"].isin(classes)]
    ct = pd.crosstab(sub["gold"], sub["pred"]).reindex(index=classes)
    ct = ct.reindex(columns=classes, fill_value=0)             # square over gold classes
    norm = ct.div(df["gold"].value_counts().reindex(classes), axis=0)  # P(pred|gold), full denom

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set(xlabel="predicted subclass", ylabel="journal-gold subclass",
           title="Confusion: P(predicted | journal-gold), top subclasses")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = norm.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > .5 else "#333")
    fig.colorbar(im, label="fraction of gold class", shrink=.8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
