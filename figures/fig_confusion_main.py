"""
figures/fig_confusion_main.py  —  G3: main-class (compact) confusion vs journal-gold.

Run on the SERVER (uses the full classified corpus + journal-gold):
    python figures/fig_confusion_main.py \
        --input classified/classified_v3_300k.parquet \
        --gold  reports/journal_validation_v3_300k/journal_gold_TK.csv \
        --out   reports/thesis_figures/fig_confusion_main.png

Main class = first letter of the LCC code. Row-normalised P(predicted | journal-gold).
A legible companion to the dense 18x18 subclass matrix: it directly visualises the
headline "main-class 0.80 strict" result.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--gold", default="reports/journal_validation_v3_300k/journal_gold_TK.csv")
    ap.add_argument("--out", default="reports/thesis_figures/fig_confusion_main.png")
    ap.add_argument("--pred-col", default="pred_lcc",
                    help="prediction column; main class = its first letter")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    gold = pd.read_csv(args.gold, dtype=str).fillna("")
    gold["gold_lcc_sub"] = gold["gold_lcc_sub"].str.strip().str.upper()
    gold = gold[gold["gold_lcc_sub"] != ""]
    gmap = dict(zip(gold["journal"].map(_clean), gold["gold_lcc_sub"]))

    pf = pq.ParquetFile(args.input)
    parts = []
    for b in pf.iter_batches(batch_size=200000, columns=["journal", args.pred_col]):
        d = b.to_pandas()
        g = d["journal"].astype(str).map(_clean).map(gmap)
        m = g.notna()
        parts.append(pd.DataFrame({
            "gold": g[m].astype(str).str[0].values,                       # main = 1st letter
            "pred": d[args.pred_col][m].astype(str).str[0].values}))
    df = pd.concat(parts)
    print(f"  {len(df):,} papers matched to gold journals")

    classes = sorted(df["gold"].value_counts().index.tolist())
    ct = pd.crosstab(df["gold"], df["pred"]).reindex(index=classes)
    ct = ct.reindex(columns=classes, fill_value=0)
    norm = ct.div(df["gold"].value_counts().reindex(classes), axis=0)

    diag = np.diag(norm.reindex(index=classes, columns=classes).values)
    overall = float(np.average(diag, weights=df["gold"].value_counts().reindex(classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set(xlabel="predicted main class", ylabel="journal-gold main class",
           title=f"Main-class confusion: P(predicted | journal-gold)\n"
                 f"weighted diagonal (recall) = {overall:.2f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = norm.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > .5 else "#333")
    fig.colorbar(im, label="fraction of gold class", shrink=.8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
