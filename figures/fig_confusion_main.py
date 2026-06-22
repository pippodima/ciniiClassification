"""
figures/fig_confusion_main.py  —  G3: main-class confusion vs journal-gold.

Run on the SERVER (uses the full classified corpus + journal-gold):
    python figures/fig_confusion_main.py \
        --input classified/classified_v3_300k.parquet \
        --gold  reports/journal_validation_v3_300k/journal_gold_TK.csv \
        --out   reports/thesis_figures/fig_confusion_main.png

Main class = first letter of the LCC code. Row-normalised P(predicted | journal-gold).
A legible companion to the dense subclass matrix.

Rows whose main class has NO in-vocabulary subclass (e.g. S = Agriculture,
V = Naval Architecture) are marked: their diagonal is structurally zero (a
vocabulary gap, not a model error), exactly as the TP/SF rows in the subclass
matrix. Two recalls are reported: over the in-vocabulary rows (matches the
thesis headline) and over all rows.

Use --exclude-vocab-gap to drop those rows entirely instead of marking them.
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
from matplotlib.patches import Rectangle

# main-class full names (LCC)
MAIN = {
    "A": "General", "B": "Phil./Psych.", "C": "Aux. Hist.", "D": "World Hist.",
    "E": "Amer. Hist.", "F": "Amer. Hist.", "G": "Geography", "H": "Social Sci.",
    "J": "Pol. Sci.", "K": "Law", "L": "Education", "M": "Music", "N": "Fine Arts",
    "P": "Lang./Lit.", "Q": "Science", "R": "Medicine", "S": "Agriculture",
    "T": "Technology", "U": "Mil. Sci.", "V": "Naval Sci.", "Z": "Bibliography",
}


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
    ap.add_argument("--exclude-vocab-gap", action="store_true",
                    help="drop gold main classes the model cannot predict")
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
            "gold": g[m].astype(str).str[0].values,                 # main = 1st letter
            "pred": d[args.pred_col][m].astype(str).str[0].values}))
    df = pd.concat(parts, ignore_index=True)
    print(f"  {len(df):,} papers matched to gold journals")

    pred_mains = set(df["pred"].unique())                            # what the model can output
    classes = sorted(df["gold"].value_counts().index.tolist())
    gap = [c for c in classes if c not in pred_mains]                # unwinnable (no in-vocab class)

    if args.exclude_vocab_gap:
        classes = [c for c in classes if c not in gap]
        df = df[df["gold"].isin(classes)]
        gap = []

    supp = df["gold"].value_counts().reindex(classes)
    # square matrix over the gold main classes (off-diagonal mass to non-gold
    # mains is negligible and only adds empty columns)
    ct = pd.crosstab(df["gold"], df["pred"]).reindex(index=classes)
    ct = ct.reindex(columns=classes, fill_value=0)
    norm = ct.div(supp, axis=0)

    # recalls
    diag = np.diag(norm.values)
    w = supp.values
    rec_all = float(np.average(diag, weights=w))
    inv = [i for i, c in enumerate(classes) if c not in gap]
    rec_inv = float(np.average(diag[inv], weights=w[inv])) if inv else rec_all

    # ---- plot ----
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.8 + 0.95 * n, 1.8 + 0.85 * n))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)

    dagger = {c: " †" for c in gap}
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{c}\n{MAIN.get(c, '')}" for c in classes], fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c} — {MAIN.get(c, '')}{dagger.get(c, '')}" for c in classes],
                       fontsize=9)
    for lbl, c in zip(ax.get_yticklabels(), classes):
        if c in gap:
            lbl.set_color("#c07a2b")

    for i in range(n):
        for j in range(n):
            v = norm.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > .55 else "#333")

    # highlight the diagonal cells (correct = recall)
    for i in range(n):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                               edgecolor="#d9534f", lw=1.8))
    # dashed outline on vocabulary-gap rows (structural zero diagonal)
    for c in gap:
        i = classes.index(c)
        ax.add_patch(Rectangle((-.5, i - .5), n, 1, fill=False,
                               edgecolor="#c07a2b", lw=1.4, ls=(0, (4, 3))))

    ax.set_xlabel("predicted main class")
    ax.set_ylabel("journal-gold main class")
    ax.set_title("Main-class confusion vs. journal-gold  "
                 r"$P(\mathrm{predicted}\mid\mathrm{gold})$", pad=12)

    note = f"in-vocabulary recall = {rec_inv:.2f}"
    if gap:
        names = ", ".join(f"{c} {MAIN.get(c, '')}" for c in gap)
        note += (f"      † no in-vocabulary subclass (vocabulary gap): {names}"
                 f"  —  {rec_all:.2f} including these rows")
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8.5, color="#444")

    fig.colorbar(im, ax=ax, label="fraction of gold class", shrink=.85)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"  in-vocab recall {rec_inv:.3f} | all-rows {rec_all:.3f} | gap={gap}")
    print(f"  ✅ {args.out}")


if __name__ == "__main__":
    main()
